from pipeline.try_gpt2 import *


def run_eval(file):
    with open("start.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        starts = text.split("^^^^^^^^^")[:-1]

    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        scenes = text.split("-------------")[:-1]

    num_of_speeches = 0
    num_of_sent = 0
    for start, scene in zip(starts, scenes):
        generated = scene.replace(start, '')
        for sent in generated.split('\n'):
            if contain_speakers(sent) or only_speakers(sent):
                num_of_speeches += 1
            num_of_sent += 1
    return num_of_speeches / len(scenes), num_of_sent/num_of_speeches


print(run_eval("tfidf_outline.txt"))
