from summa.summarizer import summarize
import os
print(os.getcwd())
from utils.utils import parse_corp
import argparse


# TODO: 200 here is number of words, not sub-words

def outline_by_textrank(text, words=200, name_length=30, language='german', split=True):
    characters = get_speakers(text, name_length)
    text_without_speaker = text
    # first remove speaker names from the raw text
    # improve the quality of extraction
    for character in characters:
        text_without_speaker = text_without_speaker.replace(character, "")
    # extract using textrank summarization

    outline = summarize(text_without_speaker, words=words, language=language, split=split)
    '''if outline_wo_speaker:
        return ' '.join(outline)'''
    outline_with_speaker = ''

    # search for each outline sentence in the raw text and add the speaker names
    speaker = ''
    same_speaker = 0
    for sent in outline:
        for line in text.split('\n'):
            for character in characters:
                if line.find(character) == 0:
                    speaker = character
                    same_speaker = 0
                    break
            if sent in line:  # TODO: 一句一行后要不要改为==/还用in
                if same_speaker == 0:
                    sent = speaker + '\n' + sent
                    same_speaker = 1
                outline_with_speaker += (sent + '\n')
                break
    return outline_with_speaker, ' '.join(outline)


def get_speakers(scene, name_length):
    speakers = []
    for line in scene.split('\n'):
        if len(line.split(':')) > 1:
            name = line.split(':')[0]
            if len(name) <= name_length:
                speakers.append(f'{name}:')            
    return list(set(speakers))


if __name__ == '__main__':
    '''file = '../test_folder/line_scene1.txt'
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        outline = outline_by_textrank(text, words=200)
        # for sent in outline:
        #    print(sent)
        print(outline)'''
    # run: python3 pipeline/outline_extraction_text_rank.py dracor test_folder/parsed_dracor_no_names_full
    # dracor 'test_folder/parsed_dracor_no_names_full'
    # dta 'outline_extraction/parsed_dta'
    parser = argparse.ArgumentParser(description='Extract outlines from draco_texts.')
    parser.add_argument('in_dir', type=str, help='Path to the corpus with plays.')
    parser.add_argument('out_dir', type=str, help='Path to directory to write outlines')

    args = parser.parse_args()
    import os

    print(os.getcwd())
    # parse corpora
    """
    txts, idx = parse_corp(args.in_file, keep_speakers=True, corpus=args.action,
                           add_newline=True)
    outline_out = 'outline_extraction/test_folder/outline/dracor/'
    outlines = []
    for i, x in enumerate(txts[0:2]):
        outl = outline_by_textrank(x)
        # write_outline(outline_out + str(idx[i]) + '.txt', outl)
        # outl = ' '.join(outl)
        print('Outline:', outl)
        # outlines.append(outl)
    """
    in_dir = "data_split"
    out_dir = "outline_split"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    datasets = ['train', 'dev', 'test']
    for dataset in datasets:
        folder = os.path.join(in_dir, dataset)
        write_folder = os.path.join(out_dir, dataset)
        if not os.path.isdir(write_folder):
            os.mkdir(write_folder)
        for i, file in enumerate(os.listdir(folder)):
            read_path = os.path.join(folder, file)
            with open(read_path, 'r', encoding='utf-8') as f:
                txt = f.read()
            outl = outline_by_textrank(txt)
            write_path = os.path.join(write_folder, file)
            with open(write_path, 'w', encoding='utf-8') as f:
                f.write(''.join(outl))
