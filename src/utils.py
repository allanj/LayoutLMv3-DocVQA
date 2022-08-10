import json
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import collections
import logging
from tqdm import tqdm
import os
import textdistance as td

logger = logging.getLogger(__name__)

def bbox_string(box, width, length):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / length)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / length))
    ]



def write_data(file:str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)

def read_data(file:str):
    with open(file, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)
    return data

def get_optimizers(model: nn.Module, learning_rate: float, num_training_steps: int, weight_decay:float = 0.01,
                   warmup_step: int = -1, eps:float = 1e-8) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    # no_decay = ["b ias", "LayerNorm.weight", 'LayerNorm.bias']
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps) # , correct_bias=False)
    warmup_step = warmup_step if warmup_step >= 0 else int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step: step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def postprocess_qa_predictions(
    metadata,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    NOTE: Adapted from from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/utils_qa.py
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(metadata):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(metadata)} samples.")


    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_predictions_list = []
    all_nbest_json = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, current_meta in enumerate(metadata):

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        # We grab the predictions of the model for this feature.
        start_logits = all_start_logits[example_index]
        end_logits = all_end_logits[example_index]
        # This is what will allow us to map some the positions in our logits to span of texts in the original
        # context.
        subword_idx2word_idx = current_meta["subword_idx2word_idx"]
        num_question_tokens = current_meta["num_question_tokens"]

        # Go through all possibilities for the `n_best_size` greater start and end logits.
        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index < num_question_tokens or ## because we do not want to consider answers that are
                    end_index < num_question_tokens or  ## part of the question.
                    start_index >= len(subword_idx2word_idx)
                    or end_index >= len(subword_idx2word_idx)
                    or subword_idx2word_idx[start_index] is None
                    or subword_idx2word_idx[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue

                prelim_predictions.append(
                    {
                        "word_ids": (subword_idx2word_idx[start_index], subword_idx2word_idx[end_index]),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    }
                )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = current_meta["words"]
        for pred in predictions:
            offsets = pred.pop("word_ids")
            pred["text"] = ' '.join(context[offsets[0] : offsets[1] + 1])

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        all_predictions[current_meta["questionId"]] = predictions[0]["text"]
        all_predictions_list.append({
            "questionId": current_meta["questionId"],
            "answer": predictions[0]["text"],
        })
        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[current_meta["questionId"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]


    return all_predictions, all_predictions_list


def anls_metric_str(predictions: List[str], gold_labels: List[List[str]], tau=0.5, rank=0):
    res = []
    """
    predictions: List[List[int]]
    gold_labels: List[List[List[int]]]: each instances probably have multiple gold labels.
    """
    for i, (pred, golds) in enumerate(zip(predictions, gold_labels)):
        max_s = 0
        for gold in golds:
            dis = td.levenshtein.distance(pred.lower(), gold.lower())
            max_len = max(len(pred), len(gold))
            if max_len == 0:
                s = 0
            else:
                nl = dis / max_len
                s = 1-nl if nl < tau else 0
            max_s = max(s, max_s)
        res.append(max_s)
    return res, sum(res)/len(res)
