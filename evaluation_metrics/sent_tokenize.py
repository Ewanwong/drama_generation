"""
desired format:

character A:
sent 1
sent 2
character B:
sent 3
sent 4

"""

import nltk
import os

prefix = 'test_folder/draco_texts/'
write_prefix = 'test_folder/texts_by_line/'
file_names = os.listdir(prefix)
for file in file_names:

    lines = []

    with open(prefix+file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines.append(line)

    sents = []
    for line in lines:
        sent = nltk.tokenize.sent_tokenize(line)
        sents += sent
    with open(write_prefix+file, 'w', encoding='utf-8') as f:
        for sent in sents:
            f.write(sent+'\n')
