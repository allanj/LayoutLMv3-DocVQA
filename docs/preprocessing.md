
## Preprocessing: Extracting Answer Spans

After the dataset preprocessing, we should end up a Huggingface datasets cached file with the following keys:
```
DatasetDict({
    train: Dataset({
        features: ['questionId', 'question', 'image', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'data_split', 'ocr_text', 'words', 'layout', 'answers'],
        num_rows: 39463
    })
    val: Dataset({
        features: ['questionId', 'question', 'image', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'data_split', 'ocr_text', 'words', 'layout', 'answers'],
        num_rows: 5349
    })
    test: Dataset({
        features: ['questionId', 'question', 'image', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'data_split', 'ocr_text', 'words', 'layout', 'answers'],
        num_rows: 5188
    })
})
```
Basically, we try to reserve all the information here.
The final statistics about the number of spans founded is as follows:

| Train / #found spans / #not found | Validation  / #found spans / #not found | Test  |
|:---------------------------------:|:---------------------------------------:|:-----:|
|      39,643 / 36,759 / 2,704      |           5,349 / 4,950 / 399           | 5,188 |

Note that the `answers` key is a dummy key in the `test` set as the answers are not provided.

### Running the Preprocessing script.

You can actually modify the `all_lowercase` in `extract_spans.py` to change the case of the all the text.
Here, we just follow this [docvqa repo](https://github.com/anisha2102/docvqa) to use the `all_lowercase` as `True` in 
my experiments.



### OCR 
We are just using the OCR provided by the docvqa dataset. However, as mentioned in this [issue](https://github.com/microsoft/unilm/issues/799),
the OCR might affect the performance.