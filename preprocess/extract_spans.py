from tqdm import tqdm
import editdistance
from collections import defaultdict
from datasets import DatasetDict, Dataset
from src.utils import read_data
from preprocess.utils import get_answer_indices,clean_text



def extract_start_end_index(current_answers, words):
    ## extracting the start, end index
    processed_answers = []
    ## remove duplicates because of the case of multiple answers
    current_answers = list(set(current_answers))
    for ans_index in range(len(current_answers)):
        current_ans = current_answers[ans_index]
        start_index, end_index, extracted_answer = get_answer_indices(words, current_answers[ans_index])
        ans = current_ans.lower()
        extracted_answer = clean_text(extracted_answer)
        ans = clean_text(ans)
        dist = editdistance.eval(extracted_answer.replace(' ', ''),
                                 ans.replace(' ', '')) if extracted_answer != None else 1000
        if start_index == -1:
            start_index = -1
            end_index = -1
            extracted_answer = ""
        if dist > 5:
            start_index = -1
            end_index = -1
        if start_index == -1 or len(extracted_answer) > 150 or extracted_answer == "":
            start_index = -1
            end_index = -1
            extracted_answer = ""
        processed_answers.append({
            "start_word_position": start_index,
            "end_word_position": end_index,
            "gold_answer": current_ans,
            "extracted_answer": extracted_answer})
    return processed_answers

def convert_docvqa_to_cache(train_file, val_file, test_file, lowercase:bool) -> DatasetDict:
    data_dict = {}
    for file in [train_file, val_file, test_file]:
        new_all_data = defaultdict(list)
        data = read_data(file)
        split = data["dataset_split"]
        objs = data['data']
        num_answer_span_found = 0
        for obj in tqdm(objs, desc="reading {}".format(split), total=len(objs)):
            new_answers = None
            for key in obj.keys():
                if key == "question":
                    new_all_data[key].append(obj[key].lower() if lowercase else obj[key])
                elif key == "answers":
                    answers = obj[key]
                    new_answers = []
                    for ans in answers:
                        new_answers.append(ans.lower() if lowercase else ans)
                    ## not added yet, later after processing, we add start and end as well.
                    # new_all_data[key].append(new_answers)
                else:
                    new_all_data[key].append(obj[key])
            ocr_file = f"data/docvqa/{split}/ocr_results/{obj['ucsf_document_id']}_{obj['ucsf_document_page_no']}.json"
            ocr_data = read_data(ocr_file)
            assert len(ocr_data['recognitionResults']) == 1
            all_text = ' '.join([line['text'] for line in ocr_data['recognitionResults'][0]['lines']])
            text, layout = [], []
            for line in ocr_data["recognitionResults"][0]["lines"]:
                for word in line["words"]:
                    x1, y1, x2, y2, x3, y3, x4, y4 = word['boundingBox']
                    new_x1 = min([x1, x2, x3, x4])
                    new_x2 = max([x1, x2, x3, x4])
                    new_y1 = min([y1, y2, y3, y4])
                    new_y2 = max([y1, y2, y3, y4])
                    if word["text"].startswith("http") or word["text"] == "":
                        continue
                    text.append(word["text"].lower() if lowercase else word["text"])
                    layout.append([new_x1, new_y1, new_x2, new_y2])
            new_all_data['ocr_text'].append(all_text)
            new_all_data['words'].append(text)
            new_all_data['layout'].append(layout)
            if new_answers is not None:
                processed_answers = extract_start_end_index(new_answers, text)
            else:
                processed_answers = [{
                    "start_word_position": -1,
                    "end_word_position": -1,
                    "gold_answer": "<NO_GOLD_ANSWER>",
                    "extracted_answer": ""}]
            new_all_data['answers'].append(processed_answers)

            ## Note: just to count the stat
            for ans in processed_answers:
                if ans['start_word_position'] != -1:
                    num_answer_span_found += 1
                    break
        total_num = len(new_all_data["questionId"])
        print(f"{split} has {total_num} questions, "
              f"extractive answer found: {num_answer_span_found} "
              f"answer not found: {total_num - num_answer_span_found}", flush=True)
        data_dict[split] = Dataset.from_dict(new_all_data)
    all_data = DatasetDict(data_dict)
    return all_data



if __name__ == '__main__':
    all_lowercase = True
    dataset = convert_docvqa_to_cache("data/docvqa/train/train_v1.0.json",
                                      "data/docvqa/val/val_v1.0.json",
                                      "data/docvqa/test/test_v1.0.json",
                                      lowercase=all_lowercase)
    cached_filename = "data/docvqa_cached_extractive_uncased" if all_lowercase else "data/docvqa_cached_extractive_cased"
    dataset.save_to_disk(cached_filename)
