def run_eval(file):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()

    word_count = 0
    sent_count = 0
    for sent in text.split("\n"):
        if ':' in sent:
            continue
        else:
            sent_count += 1
            word_count += len(sent.split(' '))

    return word_count / sent_count


avg_len = run_eval("tfidf_outline.txt")
print(avg_len)
