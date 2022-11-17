import nltk

n = 1

def run_eval(n, file, part):
    def get_ngram(n, text):
        ngram_list = []
        sents = text.split('\n')
        for sent in sents:
            tokens = nltk.word_tokenize(sent, language='german')
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n):
                ngram = tuple(tokens[i:i + n])
                ngram_list.append(ngram)
        unique_ngrams = list(set(ngram_list))
        if len(ngram_list) == 0:
            return unique_ngrams, 0
        return unique_ngrams, len(unique_ngrams)/len(ngram_list)


    with open("start_model4.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        starts = text.split("^^^^^^^^^")[:-1]

    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        scenes = text.split("-------------")[:-1]

    generates = []
    for start, scene in zip(starts, scenes):
        generated = scene.replace(start, '')
        len_gen = len(generated)
        if part == "all":
            generates.append(generated)
        elif part == "first_half":
            generates.append(generated[:len_gen//2])
        elif part == "second_half":
            generates.append(generated[len_gen//2:])


    assert len(starts) == len(generates)
    recalls = []
    precisions = []
    f1s = []
    n_reps = []
    for i in range(len(starts)):
        start = starts[i]
        generated = generates[i]
        start_ngram, _ = get_ngram(n, start)
        generated_ngram, n_rep = get_ngram(n, generated)
        n_reps.append(n_rep)
        if len(start_ngram) * len(generated_ngram) == 0:
            continue
        union = list(set(start_ngram + generated_ngram))
        overlap = len(start_ngram) + len(generated_ngram) - len(union)
        recall = overlap / len(start_ngram)
        precision = overlap / len(generated_ngram)
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precision / (recall+precision)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    print("average recall:")
    print(sum(recalls) / len(recalls))
    print("average precision:")
    print(sum(precisions) / len(precisions))
    print("average f1:")
    print(sum(f1s)/len(f1s))

    print("average unique ngram proportion:")
    print(sum(n_reps)/len(n_reps))


if __name__ == "__main__":
    run_eval(3, "tfidf_outline.txt", "second_half")