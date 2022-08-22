import editdistance
from typing import List
import textdistance as td

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


def anls_metric_str(predictions: List[List[str]], gold_labels: List[List[str]], tau=0.5, rank=0):
    res = []
    """
    predictions: List[List[int]]
    gold_labels: List[List[List[int]]]: each instances probably have multiple gold labels.
    """
    for i, (preds, golds) in enumerate(zip(predictions, gold_labels)):
        max_s = 0
        for pred in preds:
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

from sacremoses import MosesDetokenizer

def fuzzy_diff(s1, s2):
    return (editdistance.eval(s1, s2) / ((len(s1) + len(s2)) / 2))

def fuzzy(s1, s2, threshold=0.2):
    return (editdistance.eval(s1, s2) / ((len(s1) + len(s2)) / 2)) < threshold

def better_subfinder(words_list, answer_query, try_hard=True):
    matches = []
    start_indices = []
    end_indices = []

    detokenizer = MosesDetokenizer(lang="en")

    # first try dumber, faster method, but this has false negatives
    answer_list = answer_query.split()
    for idx, i in enumerate(range(len(words_list))):
        # if (
        #     words_list[i] == answer_list[0]
        #     and words_list[i : i + len(answer_list)] == answer_list
        # ):
        if len(words_list[i: i + len(answer_list)]) == len(answer_list) and all(
                fuzzy(words_list[i + j], answer_list[j]) for j in range(len(answer_list))
        ):
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)

    if matches:
        return matches[0], start_indices[0], end_indices[0]

    if not try_hard:
        # fail
        return None, 0, 0

    # if that failed, use our stronger method to find missed matches
    smart_matches = []
    for start_pos in range(len(words_list)):
        for end_pos in range(start_pos, len(words_list)):
            # use a length heuristic
            n_pieces = end_pos - start_pos + 1

            # check that the n pieces is close to the length of the answer list
            if (
                    abs(n_pieces - len(answer_list)) > 5
                    # or n_pieces < len(answer_list) / 2
                    or n_pieces > len(answer_list) * 2 + 2
            ):
                # print(f'  discarding', n_pieces, len(answer_list))
                # print(f'  discarding [{start_pos}:{end_pos}]', n_pieces, len(answer_list))
                continue

            piece = words_list[start_pos:end_pos + 1]
            # print('checking piece:', piece)

            # try to detokenize
            detok_variants = []
            detok_variants.append(' '.join(piece))
            detok_variants.append(detokenizer.detokenize(piece))
            detok_variants.append(''.join(piece))

            for detok_variant in detok_variants:
                # check if this piece is close to the answer
                diff = fuzzy_diff(detok_variant, answer_query)

                if diff == 0:
                    break  # perfect match, no need to continue

                # print(' detok piece:', detok_piece, 'diff:', diff)
                if (
                        detok_variant == answer_query
                        or diff <= 0.25
                        or answer_query in detok_variant
                ):
                    # print(f'  approx match: {detok_variant}, diff: {diff}')
                    smart_matches.append((piece, diff, start_pos, end_pos))
                    break

    if smart_matches:
        # sort smart matches by diff
        best_match = sorted(smart_matches, key=lambda x: x[1])[0]

        return best_match[0], best_match[2], best_match[3]

    # fail
    return None, 0, 0


def locate_encoded_answer(encoding, batch_index, word_idx_start, word_idx_end):
    sequence_ids = encoding.sequence_ids(batch_index)
    # Start token index of the current span in the text.
    token_start_index = 0
    # skip <pad> tokens
    while sequence_ids[token_start_index] != 1:
        token_start_index += 1

    # End token index of the current span in the text.
    token_end_index = len(encoding.input_ids[batch_index]) - 1
    # skip <pad> tokens
    while sequence_ids[token_end_index] != 1:
        token_end_index -= 1

    word_ids = encoding.word_ids(batch_index)[
               token_start_index: token_end_index + 1
               ]
    print(
        "sliced word ids from",
        token_start_index,
        "to",
        token_end_index + 1,
        "out of",
        0,
        len(encoding.word_ids(batch_index)),
    )
    # print('trying to match start and end tokens:', word_ids, word_idx_start, word_idx_end)
    # decoded_words = tokenizer.decode(
    #     encoding.input_ids[batch_index][token_start_index : token_end_index + 1]
    # )
    # print('decoded_words:', decoded_words)
    # all_words = tokenizer.decode(encoding.input_ids[batch_index])
    # print('all_words:', all_words)
    found_start = False
    found_end = False
    for id in word_ids:
        if id == word_idx_start:
            print(" start:", token_start_index)
            found_start = True
            break
        else:
            token_start_index += 1
            # print(' start id did not match:', id, word_idx_start)

    for id in word_ids[::-1]:
        if id == word_idx_end:
            print(" end:", token_end_index)
            found_end = True
            break
        else:
            token_end_index -= 1
            # print(' end id did not match:', id, word_idx_end)

    if not found_start or not found_end:
        return -1, -1

    # success
    return token_start_index, token_end_index
