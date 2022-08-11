import editdistance

def bbox_string(box, width, length):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / length)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / length))
    ]


def clean_text(text):
    replace_chars = ',.;:()-/$%&*'
    for j in replace_chars:
        if text is not None:
            text = text.replace(j, '')
    return text


def harsh_find(answer_tokens, words):
    answer_raw = ''.join(answer_tokens)
    answer = ' '.join(answer_tokens)
    if len(answer_tokens) == 1:
        for (ind, w) in enumerate(words):
            dist = 0 if len(answer) < 5 else 1
            if editdistance.eval(answer, w) <= dist:
                start_index = end_index = ind
                return start_index, end_index, w
    for (ind, w) in enumerate(words):
        if answer_raw.startswith(w):  # Looks like words are split
            for inc in range(1, 30):
                if ind + inc >= len(words):
                    break
                w = w + words[ind + inc]
                if len(answer_raw) >= 5:
                    dist = 1
                else:
                    dist = 0
                start_index = ind
                end_index = ind + inc
                ext_list = words[start_index:end_index + 1]
                extracted_answer = ' '.join(ext_list)

                if editdistance.eval(answer.replace(' ', ''), extracted_answer.replace(' ', '')) <= dist:
                    return start_index, end_index, extracted_answer
    return reverse_harsh_find(answer_tokens, words)


def reverse_harsh_find(answer_tokens, words):
    answer_raw = ''.join(answer_tokens)
    answer = ''.join(answer_tokens)
    for (ind, w) in enumerate(words):
        if answer_raw.endswith(w):  # Looks like words are split
            for inc in range(1, 30):
                if ind - inc < 0:
                    break
                w = words[ind - inc] + w
                if len(answer_raw) >= 15:
                    dist = 3
                elif len(answer_raw) >= 5:
                    dist = 1
                else:
                    dist = 0
                start_index = ind - inc
                end_index = ind
                ext_list = words[start_index:end_index + 1]
                extracted_answer = ' '.join(ext_list)

                if editdistance.eval(answer.replace(' ', ''), extracted_answer.replace(' ', '')) <= dist:
                    return start_index, end_index, extracted_answer
    return None, None, None


def get_answer_indices(words, answer):
    answer_tokens = answer.split()
    end_index = None
    start_index = None
    words = [clean_text(x) for x in words]
    answer_tokens = [clean_text(x) for x in answer_tokens]
    answer = ' '.join(answer_tokens)

    if answer_tokens[0] in words:
        start_index = words.index(answer_tokens[0])
    if answer_tokens[-1] in words:
        end_index = words.index(answer_tokens[-1])
    if start_index is not None and end_index is not None:
        if start_index > end_index:
            if answer_tokens[-1] in words[start_index:]:
                end_index = words[start_index:].index(answer_tokens[-1])
                end_index += start_index
            else:
                # Last try
                start_index, end_index, extracted_answer = harsh_find(answer_tokens, words)
                return start_index, end_index, extracted_answer

        assert start_index <= end_index
        extracted_answer = ' '.join(words[start_index:end_index + 1])
        if answer.replace(' ', '') != extracted_answer.replace(' ', ''):
            start_index, end_index, extracted_answer = harsh_find(answer_tokens, words)
            return start_index, end_index, extracted_answer
        else:
            return start_index, end_index, extracted_answer

        return None, None, None
    else:
        answer_raw = ''.join(answer_tokens)
        start_index, end_index, extracted_answer = harsh_find(answer_tokens, words)
        return start_index, end_index, extracted_answer
