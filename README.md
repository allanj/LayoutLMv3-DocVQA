# Fine-tuning LayoutLMv3 on DocVQA

We try to reproduce the experiments for fine-tuning LayoutLMv3 on [DocVQA](https://www.docvqa.org/datasets/docvqa) using both 
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
   Then you will get something called ``
   More details about the statistics after preprocessing, Check out [here](/docs/preprocess.md).