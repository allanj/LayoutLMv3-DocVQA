from typing import Dict, Optional, List
import torch.nn as nn
import os
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor, LayoutLMv3Processor
from dataclasses import dataclass
from PIL import Image
import torch
import cv2
from src.utils import bbox_string

def get_subword_start_end(word_start, word_end, subword_idx2word_idx, sequence_ids):
    ## find the separator between the questions and the text
    start_of_context = -1
    for i in range(len(sequence_ids)):
        if sequence_ids[i] == 1:
            start_of_context = i
            break
    num_question_tokens = start_of_context
    assert start_of_context != -1, "Could not find the start of the context"
    subword_start = -1
    subword_end = -1
    for i in range(start_of_context, len(subword_idx2word_idx)):
        if word_start == subword_idx2word_idx[i] and subword_start == -1:
            subword_start = i
        if word_end == subword_idx2word_idx[i]:
            subword_end = i
    return subword_start, subword_end, num_question_tokens

"""
handling the out of maximum length reference:
https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
"""

def tokenize_docvqa(examples,
                    tokenizer: LayoutLMv3TokenizerFast,
                    img_dir: Dict[str, str],
                    add_metadata: bool = True,
                    use_msr_ocr: bool = False,
                    use_generation: bool = False,
                    doc_stride: int = 128): ## doc stride for sliding window, if 0, means no sliding window.

    features = {"input_ids": [], "image":[], "bbox":[], "start_positions": [], "end_positions":[],  "metadata": []}
    current_split = examples["data_split"][0]
    if use_generation:
        features["labels"] = []
    for idx, (question, image_path, words, layout) in enumerate(zip(examples["question"], examples["image"], examples["words"], examples["layout"])):
        current_metadata = {}
        file = os.path.join(img_dir[examples["data_split"][idx]], image_path)
        # img = Image.open(file).convert("RGB")
        answer_list = examples["processed_answers"][idx] if "processed_answers" in examples else []
        original_answer = examples["original_answer"][idx] if "original_answer" in examples else []
        # image_id = f"{examples['ucsf_document_id'][idx]}_{examples['ucsf_document_page_no'][idx]}"
        if len(words) == 0 and current_split == "train":
            continue
        tokenized_res = tokenizer.encode_plus(text=question, text_pair=words, boxes=layout, add_special_tokens=True,
                                              max_length=512, truncation="only_second",
                                              return_offsets_mapping=True, stride=doc_stride, return_overflowing_tokens=doc_stride>0)
        sample_mapping = tokenized_res.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_res.pop("offset_mapping")

        if use_generation:
            dummy_boxes = [[[0,0,0,0]] for _ in range(len(original_answer))]
            answer_ids = tokenizer.batch_encode_plus([[ans] for ans in original_answer], boxes = dummy_boxes,
                                                     add_special_tokens=True, max_length=100, truncation="longest_first")["input_ids"]
        else:
            answer_ids = [[0] for _ in range(len(original_answer))]
        if not use_msr_ocr:
            img = cv2.imread(file)
            height, width = img.shape[:2]

        for stride_idx, offsets in enumerate(offset_mapping):
            input_ids = tokenized_res["input_ids"][stride_idx]
            subword_idx2word_idx = tokenized_res.encodings[stride_idx].word_ids
            sequence_ids = tokenized_res.encodings[stride_idx].sequence_ids
            if current_split == "train":
                # for training, we treat instances with multiple answers as multiple instances
                for answer, label_ids in zip(answer_list, answer_ids):
                    if not use_generation:
                        if answer["start_word_position"] == -1:
                            subword_start = 0 ## just use the CLS
                            subword_end = 0
                            num_question_tokens = 0
                        else:
                            subword_start, subword_end, num_question_tokens = get_subword_start_end(answer["start_word_position"], answer["end_word_position"], subword_idx2word_idx, sequence_ids)
                            if subword_start == -1 or subword_end == -1:
                                subword_start = 0  ## just use the CLS
                                subword_end = 0
                    else:
                        features["labels"].append(label_ids)
                        subword_start = -1  ## useless as in generation
                        subword_end = -1
                        num_question_tokens = 0

                    features["image"].append(file)
                    features["input_ids"].append(input_ids)
                    boxes_norms = []
                    for box in tokenized_res["bbox"][stride_idx]:
                        box_norm = box if use_msr_ocr else bbox_string([box[0], box[1], box[2], box[3]], width, height)
                        boxes_norms.append(box_norm)
                    features["bbox"].append(boxes_norms)
                    features["start_positions"].append(subword_start)
                    features["end_positions"].append(subword_end)
                    current_metadata["original_answer"] = original_answer
                    current_metadata["question"] = question
                    current_metadata["num_question_tokens"] = num_question_tokens ## only used in testing.
                    current_metadata["words"] = words
                    current_metadata["subword_idx2word_idx"] = subword_idx2word_idx
                    current_metadata["questionId"] = examples["questionId"][idx]
                    current_metadata["data_split"] = examples["data_split"][idx]
                    features["metadata"].append(current_metadata)
                    if not add_metadata:
                        features.pop("metadata")
            else:
                # for validation and test, we treat instances with multiple answers as one instance
                # we just use the first one, and put all the others in the "metadata" field
                subword_start, subword_end = -1, -1
                for i in range(len(sequence_ids)):
                    if sequence_ids[i] == 1:
                        num_question_tokens = i
                        break
                features["image"].append(file)
                features["input_ids"].append(input_ids)
                boxes_norms = []
                for box in tokenized_res["bbox"][stride_idx]:
                    box_norm = box if use_msr_ocr else bbox_string([box[0], box[1], box[2], box[3]], width, height)
                    boxes_norms.append(box_norm)
                features["bbox"].append(boxes_norms)
                features["start_positions"].append(subword_start)
                features["end_positions"].append(subword_end)
                current_metadata["original_answer"] = original_answer
                current_metadata["question"] = question
                current_metadata["num_question_tokens"] = num_question_tokens
                current_metadata["words"] = words
                current_metadata["subword_idx2word_idx"] = subword_idx2word_idx
                current_metadata["questionId"] = examples["questionId"][idx]
                current_metadata["data_split"] = examples["data_split"][idx]
                features["metadata"].append(current_metadata)
                if not add_metadata:
                    features.pop("metadata")
    return features


@dataclass
class DocVQACollator:
    tokenizer: LayoutLMv3TokenizerFast
    feature_extractor: LayoutLMv3FeatureExtractor
    padding: bool = True
    model: Optional[nn.Module] = None

    def __call__(self, batch: List):

        labels = [feature["labels"] for feature in batch] if "labels" in batch[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            for feature in batch:
                remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["label_ids"] = feature["labels"] + remainder
                # print(feature["labels"])
                feature.pop("labels")

        for feature in batch:
            image = Image.open(feature["image"]).convert("RGB")
            vis_features = self.feature_extractor(images=image, return_tensors='np')["pixel_values"][0]
            feature['pixel_values'] = vis_features.tolist()
            if 'image' in feature: feature.pop('image')

        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            pad_to_multiple_of=None,
            return_tensors="pt",
            return_attention_mask=True
        )

         # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            batch.pop("start_positions")
            batch.pop("end_positions")
            if "label_ids" in batch:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["label_ids"])
                batch["decoder_input_ids"] = decoder_input_ids
                ## for validation, we don't have "labels
        if "label_ids" in batch:
            ## layoutlmv3 tokenizer issue, they don't allow "labels" key as a list..so we did a small trick
            batch["labels"] = batch.pop("label_ids")
        return batch




if __name__ == '__main__':
    from datasets import load_from_disk, DatasetDict
    from tqdm import tqdm
    dataset_file = 'data/docvqa_cached_extractive_all_lowercase_True_msr_True_include_test'
    # tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
    dataset = load_from_disk(dataset_file)
    # dataset = DatasetDict({"train": dataset["train"], "val": dataset['val']})
    # image_dir = {"train": "data/docvqa/train", "val": "data/docvqa/val", "test": "data/docvqa/test"}
    # use_msr = "msr_True" in dataset_file
    # tokenized = dataset.map(tokenize_docvqa,
    #                         fn_kwargs={"tokenizer": tokenizer,
    #                                    "img_dir": image_dir,
    #                                    "use_msr_ocr": use_msr,
    #                                    "use_generation": False,
    #                                    "doc_stride": 128},
    #                         batched=True, num_proc=8,
    #                         load_from_cache_file=True)
    max_answer_length = -1
    for obj in tqdm(dataset["train"], total=len(dataset["train"]), desc="train"):
        for ans in obj["original_answer"]:
            if len(ans.split()) > max_answer_length:
                max_answer_length = len(ans)
    print(f"maximum number of answer words: {max_answer_length}")
    # feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)
    # collator = DocVQACollator(tokenizer, feature_extractor)
    # loader = DataLoader(new_eval_dataset.remove_columns("metadata"), batch_size=3, collate_fn=collator, num_workers=1)
    # for batch in loader:
    #     print(batch.input_ids)