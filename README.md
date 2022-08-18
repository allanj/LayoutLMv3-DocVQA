# Fine-tuning LayoutLMv3 on DocVQA

We try to reproduce the experiments for fine-tuning [LayoutLMv3](https://arxiv.org/abs/2204.08387) on [DocVQA](https://www.docvqa.org/datasets/docvqa) using both 
extractive and abstractive approach.

I try to present every single detail within this repository. Note, this is not official codebase from LayoutLMv3.

__Work In Progress__


## Install Requirements
```pip3 install -r requirements.txt```

## Dataset Preprocessing
Some of the code in this repository is adapted from this [docvqa repo](https://github.com/anisha2102/docvqa) 
which works on "_LayoutLMv1 for DocVQA_".

Note that the test set from the docvqa repo does not come with the ground-truth answers.

1. Download the dataset from the [DocVQA Website](https://www.docvqa.org/datasets/docvqa) and put `docvqa` folder under `data` folder.
2. Run the following command to create the huggingface dataset:
    ```
    python3 -m preprocess.extract_spans
    ```
   Then you will get a processed called `docvqa_cached_extractive_all_lowercase_True_msr_True`
   More details about the statistics after preprocessing, Check out [here](/docs/preprocess.md).
   The final statistics about the number of spans founded is as follows:

   | Train / #found spans / #not found | Validation  / #found spans / #not found | Test  |
   |:---------------------------------:|:---------------------------------------:|:-----:|
   |      39,643 / 36,759 / 2,704      |           5,349 / 4,950 / 399           | 5,188 |
   
   __NOTE__: The microsft READ API for OCR is not available. Please contact me if you want to use this dataset. (Thanks @redthing1 giving me the access.)

## Usage
1. Run `accelerate config` to configrate your distributed training environment and run the experiments by
   ```
   accelerate launch docvqa_main.py --use_generation=0
   ```
   Set `use_generation` to 1 if you want to use the generation model.   

   My distributed training environment: 6 GPUs
   
## Current Performance (Improving :rocket:)
|             Model              |  Preprocessing   |     OCR Engine     |   Validation ANLS   |
|:------------------------------:|:----------------:|:-----:|:-------------------:|
|        LayoutLMv3-base         | lowercase inputs |      built-in      |        68.5%        |
|        LayoutLMv3-base         | lowercase inputs |   Microsoft READ API |        73.1%        |
|        LayoutLMv3-base         |  original cased  |   Microsoft READ API |        72.7%        |
| LayoutLMv3-base + Bart Decoder |  lowercase  |   Microsoft READ API | 72.4% (in progress) |

The performance is still far behind what is reported in the paper. 
But LayoutLMv3 paper combines `train+dev` and evaluate on test set, they achieve about 78% ANLS.

## TODO
- [X] Code for tokenization and Collating. (:white_check_mark:)
- [x] Code for Training (:white_check_mark:)
- [x] Further tune the performance by hyperparameters/casing issue (:white_check_mark:)
- [x] Add a decoder for generation (:white_check_mark:)
- [ ] Sliding window to handle the issue that the matched answers are out of the 512 tokens.
