from torch.utils.data import DataLoader
import os
from accelerate import Accelerator
from functools import partial
from tqdm import tqdm
import random
import torch
import numpy as np
import argparse
from transformers import PreTrainedModel, LayoutLMv3ForQuestionAnswering, LayoutLMv3TokenizerFast, LayoutLMv3FeatureExtractor, RobertaModel, LayoutLMv3Config, LayoutLMv3Model
from src.utils import get_optimizers, create_and_fill_np_array, write_data, anls_metric_str, postprocess_qa_predictions
from src.data.tokenization import tokenize_docvqa, DocVQACollator
from accelerate.utils import set_seed
from src.layoutlmv3_gen import LayoutLMv3ForConditionalGeneration

from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

tqdm = partial(tqdm, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not accelerator.is_local_main_process)

import logging
from datasets import load_from_disk, DatasetDict, concatenate_datasets

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument('--dataset_file', default="data/docvqa_cached_extractive_all_lowercase_True_msr_True", type=str)
    parser.add_argument("--model_folder", default="layoutlmv3-extractive-uncased", type=str)

    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"])
    parser.add_argument("--batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The peak learning rate.")
    parser.add_argument("--num_epochs", default=40, type=int)
    parser.add_argument('--seed', type=int, default=42,  help="random seed for initialization")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='gradient clipping max norm')
    parser.add_argument('--fp16', default=True, action='store_true', help="Whether to use 16-bit 32-bit training")

    parser.add_argument('--use_generation', default=0, choices=[0, 1], help="Whether to use generation to perform experiments")
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train(args,
          tokenizer: LayoutLMv3TokenizerFast,
          model: PreTrainedModel,
          train_dataloader: DataLoader,
          num_epochs: int, val_metadata,
          valid_dataloader: DataLoader = None,
          ):
    t_total = int(len(train_dataloader) * num_epochs)

    optimizer, scheduler = get_optimizers(model=model, learning_rate=args.learning_rate, num_training_steps=t_total, warmup_step=0, eps=1e-8)
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader)

    best_anls = -1
    os.makedirs(f"model_files/{args.model_folder}", exist_ok=True)  ## create model files. not raise error if exist
    os.makedirs(f"results", exist_ok=True)  ## create model files. not raise error if exist
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for iter, batch in tqdm(enumerate(train_dataloader, 1), desc="--training batch", total=len(train_dataloader)):
            with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                output = model(**batch)
                loss = output.loss
            total_loss += loss.item()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
        accelerator.print(
            f"Finish epoch: {epoch}, loss: {total_loss:.2f}, mean loss: {total_loss / len(train_dataloader):.2f}",
            flush=True)
        if valid_dataloader is not None:
            model.eval()
            anls = evaluate(args=args, tokenizer=tokenizer, valid_dataloader=valid_dataloader, model=model, metadata=val_metadata,
                            res_file = f"results/{args.model_folder}.res.json",
                            err_file = f"results/{args.model_folder}.err.json")
            if anls > best_anls:
                accelerator.print(f"[Model Info] Saving the best model... with best ANLS: {anls}")
                module = model.module if hasattr(model, 'module') else model
                os.makedirs(f"model_files/{args.model_folder}/", exist_ok=True)
                torch.save(module.state_dict(), f"model_files/{args.model_folder}/state_dict.pth")
                best_anls = anls
        accelerator.print("****Epoch Separation****")
    return model

def evaluate(args, tokenizer:LayoutLMv3TokenizerFast, valid_dataloader: DataLoader, model: PreTrainedModel, metadata, res_file=None, err_file=None):
    model.eval()
    if args.use_generation:
        all_pred_texts = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(args.fp16)):
            for index, batch in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
                generated_ids = model(**batch, is_train=False)
                generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id,
                                                                 pad_first=False)  ## 1 is pad token id
                generated_ids = accelerator.gather_for_metrics(generated_ids)
                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g
                         in
                         generated_ids]
                all_pred_texts.extend(preds)
        prediction_list = all_pred_texts
    else:
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(args.fp16)):
            for index, batch in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
                batch.start_positions = None
                batch.end_positions = None
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        eval_dataset = valid_dataloader.dataset
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction_dict, prediction_list = postprocess_qa_predictions(metadata=metadata, predictions=outputs_numpy)
        all_pred_texts = [prediction['answer'] for prediction in prediction_list]
    truth = [meta["original_answer"] for meta in metadata]
    all_anls, anls = anls_metric_str(predictions=all_pred_texts, gold_labels=truth)
    accelerator.print(f"[Info] Average Normalized Lev.S : {anls} ", flush=True)
    if res_file is not None and accelerator.is_main_process:
        accelerator.print(f"Writing results to {res_file} and {err_file}")
        write_data(data=prediction_list, file=res_file)
    return anls

def main():
    args = parse_arguments()
    set_seed(args.seed, device_specific=True)
    pretrained_model_name = 'microsoft/layoutlmv3-base'
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(pretrained_model_name)
    feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(pretrained_model_name, apply_ocr=False)
    if args.use_generation:
        model = LayoutLMv3ForConditionalGeneration(LayoutLMv3Config.from_pretrained(pretrained_model_name))
        model.layoutlmv3.encoder.load_state_dict(LayoutLMv3Model.from_pretrained(pretrained_model_name).state_dict())
        model.config.decoder_start_token_id = model.config.eos_token_id
        model.config.is_encoder_decoder = True
        old = RobertaModel.from_pretrained('roberta-base')
        model.layoutlmv3.decoder.layer.load_state_dict(old.encoder.layer.state_dict(), strict=False)
    else:
        model = LayoutLMv3ForQuestionAnswering.from_pretrained(pretrained_model_name)
    collator = DocVQACollator(tokenizer, feature_extractor, model=model)
    dataset = load_from_disk(args.dataset_file)
    # dataset = DatasetDict({"train": dataset["train"].select(range(100)), "val": dataset['val'].select(range(100))})
    dataset = DatasetDict({"train": dataset["train"], "val": dataset['val']})
    image_dir = {"train": "data/docvqa/train", "val": "data/docvqa/val", "test": "data/docvqa/test"}
    use_msr = "msr_True" in args.dataset_file
    tokenized = dataset.map(tokenize_docvqa,
                            fn_kwargs={"tokenizer": tokenizer,
                                       "img_dir": image_dir,
                                       "use_msr_ocr": use_msr,
                                       "use_generation": bool(args.use_generation)},
                            batched=True, num_proc=8,
                            load_from_cache_file=True,
                            # cache_file_names={
                            #     "train": f"{args.dataset_file}/train/cache_layoutlmv3_docvqa.arrow",
                            #     "val": f"{args.dataset_file}/val/cache_layoutlmv3_docvqa.arrow",
                            # },
                            remove_columns=dataset["val"].column_names
                          )
    train_dataloader = DataLoader(tokenized["train"].remove_columns("metadata"), batch_size=args.batch_size, shuffle=True,
                                      num_workers=5, pin_memory=True, collate_fn=collator)
    valid_dataloader = DataLoader(tokenized["val"].remove_columns("metadata"), batch_size=args.batch_size,
                                 collate_fn=collator, num_workers=5, shuffle=False)

    train(args=args,
          tokenizer=tokenizer,
          model=model,
          train_dataloader=train_dataloader,
          num_epochs=args.num_epochs,
          valid_dataloader=valid_dataloader,
          val_metadata=tokenized["val"]["metadata"])


if __name__ == "__main__":
    main()


