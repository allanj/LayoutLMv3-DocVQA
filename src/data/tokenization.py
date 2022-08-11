from typing import Dict, Optional, List
import torch.nn as nn
import os
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor, LayoutLMv3Processor
from dataclasses import dataclass
from PIL import Image
import torch
import cv2
from src.utils import bbox_string

def get_subword_start_end(word_start, word_end, subword_idx2word_idx):
    ## find the separator between the questions and the text
    start_of_context = -1
    for i in range(len(subword_idx2word_idx)):
        if subword_idx2word_idx[i] is None and subword_idx2word_idx[i+1] is None:
            start_of_context = i+2
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

def tokenize_docvqa(examples,
                    tokenizer: LayoutLMv3TokenizerFast,
                    img_dir: Dict[str, str],
                    add_metadata: bool = True,
                    combine_train_val_as_train: bool = False):
    """

    :param examples:
    :param tokenizer:
    :param max_seq_length:
    :param img_dir: {"train_dir": xxxx, "val_dir":xxx}
    :param add_metadata:
    :param shrink_vocab_mapper:
    :return:
    """
    features = {"input_ids": [], "image":[], "bbox":[], "start_positions": [], "end_positions":[],  "metadata": []}
    current_split = examples["data_split"][0]
    for idx, (question, image_path, words, layout) in enumerate(zip(examples["question"], examples["image"], examples["words"], examples["layout"])):
        current_metadata = {}
        file = os.path.join(img_dir[examples["data_split"][idx]], image_path)
        # img = Image.open(file).convert("RGB")
        answer_list = examples["processed_answers"][idx] if "processed_answers" in examples else []
        original_answer = examples["original_answer"][idx] if "original_answer" in examples else []
        image_id = f"{examples['ucsf_document_id'][idx]}_{examples['ucsf_document_page_no'][idx]}"
        if len(words) == 0 and current_split == "train":
            continue
        tokenized_res = tokenizer.encode_plus(text=question, text_pair=words, boxes=layout, add_special_tokens=True,
                                              return_tensors="pt", max_length=512, truncation="only_second",
                                              return_offsets_mapping=True)

        input_ids = tokenized_res["input_ids"][0]

        subword_idx2word_idx = tokenized_res.encodings[0].word_ids
        img = cv2.imread(file)
        height, width = img.shape[:2]
        if current_split == "train" or (current_split == "val" and combine_train_val_as_train):
            # for troaining, we treat instances with multiple answers as multiple instances
            for answer in answer_list:
                if answer["start_word_position"] == -1:
                    continue
                subword_start, subword_end, num_question_tokens = get_subword_start_end(answer["start_word_position"], answer["end_word_position"], subword_idx2word_idx)
                if subword_start == -1:
                    continue
                if subword_end == -1:
                    subword_end = 511 - 1  ## last is </s>, second last
                features["image"].append(file)
                features["input_ids"].append(input_ids)
                # features["attention_mask"].append(tokenized_res["attention_mask"])
                # features["bbox"].append(tokenized_res["bbox"][0])
                boxes_norms = []
                for box in tokenized_res["bbox"][0]:
                    box_norm = bbox_string([box[0], box[1], box[2], box[3]], width, height)
                    assert box[2] >= box[0]
                    assert box[3] >= box[1]
                    assert box_norm[2] >= box_norm[0]
                    assert box_norm[3] >= box_norm[1]
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
        else:
            # for validation and test, we treat instances with multiple answers as one instance
            # we just use the first one, and put all the others in the "metadata" field
            # find the first answer that has start and end
            final_start_word_pos = 1 ## if not found, just for nothing, because we don't use it anyway for evaluation
            final_end_word_pos = 1
            for answer in answer_list:
                if answer["start_word_position"] == -1:
                    continue
                else:
                    final_start_word_pos = answer["start_word_position"]
                    final_end_word_pos = answer["end_word_position"]
                    break
            subword_start, subword_end, num_question_tokens = get_subword_start_end(final_start_word_pos, final_end_word_pos, subword_idx2word_idx)
            if subword_end == -1:
                subword_end = 511 - 1  ## last is </s>, second last
            features["image"].append(file)
            features["input_ids"].append(input_ids)
            # features["attention_mask"].append(tokenized_res["attention_mask"])
            # features["bbox"].append(tokenized_res["bbox"][0])
            boxes_norms = []
            for box in tokenized_res["bbox"][0]:
                box_norm = bbox_string([box[0], box[1], box[2], box[3]], width, height)
                assert box[2] >= box[0]
                assert box[3] >= box[1]
                assert box_norm[2] >= box_norm[0]
                assert box_norm[3] >= box_norm[1]
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
        return batch

if __name__ == '__main__':
    from datasets import load_from_disk, DatasetDict
    from torch.utils.data import DataLoader
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
    dataset = load_from_disk('data/docvqa_cached_extractive_uncased')
    dataset = DatasetDict({"train": dataset["train"], "val": dataset['val']})
    image_dir = {"train": "data/docvqa/train", "val": "data/docvqa/val", "test": "data/docvqa/test"}
    new_eval_dataset = dataset.map(tokenize_docvqa,
                                          fn_kwargs={"tokenizer": tokenizer, "img_dir": image_dir},
                                          batched=True, num_proc=8,
                                          load_from_cache_file=False,
                                          remove_columns=dataset["val"].column_names
                                          )
    feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)
    # feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False, image_mean=[0.5, 0.5, 0.5],
    #                                                image_std=[0.5, 0.5, 0.5])
    collator = DocVQACollator(tokenizer, feature_extractor)
    # loader = DataLoader(new_eval_dataset.remove_columns("metadata"), batch_size=3, collate_fn=collator, num_workers=1)
    # for batch in loader:
    #     print(batch.input_ids)