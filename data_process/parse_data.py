import os.path
import pickle
import nltk
import random
import shutil
import json
import copy

random.seed(1)

ALLOWED_CORPORA = {
    'dracor': {'play_id_start': '$id_', 'play_id_end': '@id_', 'speaker_start': '$sp', 'speaker_end': '@sp_',
               'scene_st': '$scene', 'scene_end': '@scene'}, 'dta': {'play_id_start': '$id_',
                        'play_id_end': '@id_', 'speaker_start': '$sp', 'speaker_end': '@sp_',
                                                                     'scene_st': '$scene', 'scene_end': '@scene'}}
def fix_speakers_dta(filename:str, outfile:str):
    """
        Removes # from speaker abbreviations.
    @param filename: json file with a dict of abbrevation: speaker name mappings
    @param outfile: json file to which the fixed version should be saved
    """
    js = get_dta_speakers(filename)
    new_js = copy.deepcopy(js)
    for play, info in js.items():
        for abbrev, name in info['role'].items():
            # remove # from names
            if '#' in abbrev:
                tmp_abbrev = abbrev.replace('#', '')
                new_js[play]['role'][tmp_abbrev] = name
                del new_js[play]['role'][abbrev]
    with open(outfile, 'w', encoding='utf8') as f:
        json.dump(new_js, f, indent=4)

def get_dta_speakers(dta_names_file:str):
    """
        Read in file with abbrevation-speaker mappings.
    @param dta_names_file: name of the json file
    """
    with open(dta_names_file, 'r', encoding='utf8') as f:
        return json.load(f)

def parse_corp(filename: str, keep_speakers=False, corpus='dracor', add_newline=True, speaker_mappings=None):
    if corpus not in ALLOWED_CORPORA:
        raise ValueError(f'{corpus} is not in allowed corpora. Please choose from {ALLOWED_CORPORA.keys()}')
    if corpus == 'dta' and not speaker_mappings:
        raise ValueError('Please pass the file with dta abbreviation-speaker mappings.}')

    play_start = ALLOWED_CORPORA[corpus]['play_id_start']
    play_end = ALLOWED_CORPORA[corpus]['play_id_end']
    scene_start = ALLOWED_CORPORA[corpus]['scene_st']
    scene_end = ALLOWED_CORPORA[corpus]['scene_end']
    sp_start = ALLOWED_CORPORA[corpus]['speaker_start']
    sp_end = ALLOWED_CORPORA[corpus]['speaker_end']
    texts = []
    ids = []

    with open(filename, 'r', encoding='utf8') as f:
        scene_text = ''                                     # txt for current scene
        cur_id = ''                                         # id of current play
        scene_id = 1                                        # id of current scene
        for line in f:
            line = line.strip()  # TODO: leave \n in the scene_text?
            # start of the play: get the id of the play
            if line.startswith(play_start):
                if corpus == 'dta':
                    cur_id = line.split(play_start)[1]
                else:
                    cur_id = line
            # end of the play: reset play and scene ids
            elif line.startswith(play_end):
                cur_id = ''
                scene_id = 1
            elif line.startswith(scene_start):
                # need 'continue' so that the scene opening tag does not get added to text
                prev_speaker = ''
                continue
            # add current scene to txts
            elif line.startswith(scene_end):
                prev_speaker = ''
                if scene_text == '':
                    print(cur_id, ' is empty!')
                    print(line)
                    continue
                texts.append(scene_text)
                ids.append(cur_id + '_' + str(scene_id))            # id = textId + sceneId
                scene_text = ''
                scene_id += 1
            # get speakers if needed
            elif line.startswith(sp_start):  # $sp_#die_diener_1-1
                if keep_speakers:
                    speakers = line.split('#')[1:]
                    if len(speakers) > 1:
                        #print(speakers)
                        speaker = ''
                        for i, sp in enumerate(speakers):
                            sp = split_names(sp)

                            if i == len(speakers) - 1:
                                speaker += sp.capitalize()
                            else:
                                speaker += sp.strip().capitalize() + ', '

                        # look up abbreviations in name mappings
                        '''if corpus == 'dta' and speaker_mappings:
                            print(cur_id, speakers, line)
                            # if the combination of speakers is in the mappings dict, get the value
                            if ''.join(speakers).strip() in speaker_mappings[cur_id]['role']:
                                speakers = speaker_mappings[cur_id]['role'][''.join(speakers).strip()]
                                print(speakers)
                                # FIXME: add to speaker
                            # get abbreviated names one by one
                            else:
                                print('#############################')
                                print(speakers)
                                for s in speakers:
                                    print(cur_id, s, speaker_mappings[cur_id]['role'][s.strip()])
                                    print('#############################')
                                    if i == len(speakers) - 1:
                                        speaker += s.capitalize()
                                    else:
                                        speaker += s.strip().capitalize() + ', 

                        else:
                            for i, sp in enumerate(speakers):
                                sp = split_names(sp)

                                if i == len(speakers) - 1:
                                    speaker += sp.capitalize()
                                else:
                                    speaker += sp.strip().capitalize() + ', '
                        '''
                    # only one speaker
                    else:
                        # look up name abbreviation
                        '''print(line, speakers)
                        if corpus == 'dta' and speaker_mappings:
                            speakers = speaker_mappings[cur_id]['role'][''.join(speakers).strip()]'''
                        speaker = split_names(''.join(speakers))
                    #print("New speakers: ", speaker + ':' + '\n')
                    scene_text += speaker + ':' + '\n'
                    prev_speaker = speaker
                    continue
            # end of speech
            elif line.startswith(sp_end):
                # remove speaker if they haven't said anything
                if prev_speaker != '' and scene_text.endswith(prev_speaker + ':\n'):
                    '''print('------------------------')
                    print('Speaker without text, ', line)
                    print('orig text last 20 chars: ', scene_text[-20:])
                    print('Prev speaker:', prev_speaker, 'New text: ', scene_text[-20:], len(scene_text))'''
                    scene_text = scene_text[:scene_text.rfind(prev_speaker + ':\n')]

                if add_newline:
                    scene_text += '\n'
                else:
                    scene_text += ' '
            else:
                scene_text += line + ' '
            prev_speaker = ''

    return texts, ids


def split_names(name_str: str, split_sign='_'):
    """
        Splits speaker names on underscores
    """
    tmp = name_str.split(split_sign)
    speaker = ''
    for ix, x in enumerate(tmp):
        speaker += x.capitalize()
        if ix == len(tmp) - 1:
            continue
        speaker += ' '
    #print(speaker)
    return speaker


def load_pkl(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def sent_tokenize(text):
    sents = []
    for line in text.split('\n'):
        sent = nltk.tokenize.sent_tokenize(line)
        sents += [(s + '\n') for s in sent]
    return ''.join(sents)


def parse_corp_tokenized(filename: str, keep_speakers=False, corpus='dracor', add_newline=True, speaker_mappings=None):
    txs, ix = parse_corp(filename, keep_speakers, corpus, add_newline, speaker_mappings)
    texts = []
    for id, text in zip(ix, txs):
        texts.append(sent_tokenize(text))
    return texts, ix


def save_parsed_data(filename: str, prefix: str, keep_speakers=False, corpus='dracor', add_newline=True, separate_by_sentence=True, end_of_text=None):
    if separate_by_sentence:
        txs, ix = parse_corp_tokenized(filename, keep_speakers, corpus, add_newline)
    else:
        txs, ix = parse_corp(filename, keep_speakers, corpus, add_newline)
    if prefix[-1] != '/':
        prefix += '/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for id, text in zip(ix, txs):
        with open(prefix + id +'.txt', 'w', encoding='utf-8') as f:
            f.write(text)
            if end_of_text:
                f.write("<endoftext>")


def split_data(folder, save_path, ratio):
    drama_dict = {}  # drama_id : number of scenes
    all_scene_ids = os.listdir(folder)
    for scene_id in all_scene_ids:
        drama_id = '_'.join(scene_id.split('_')[:2])
        if drama_id not in drama_dict.keys():
            drama_dict[drama_id] = 1
        else:
            drama_dict[drama_id] += 1
    print(f"largest number of scenes in a play: {max(drama_dict.values())}")
    print(f"smallest number of scenes in a play: {min(drama_dict.values())}")
    dir = ['train', 'dev', 'test']
    for s in dir:
        os.makedirs(os.path.join(save_path, s), exist_ok=True)
    for drama_id, v in drama_dict.items():
        id_list = [i + 1 for i in range(v)]
        random.shuffle(id_list)
        train_ids, dev_ids, test_ids = split_by_ratio(id_list, ratio)
        for id in train_ids:
            scene_name = f"{drama_id}_{id}.txt"
            copy_path = os.path.join(folder, scene_name)
            paste_path = os.path.join(save_path, 'train', scene_name)
            shutil.copy(copy_path, paste_path)
        if len(dev_ids) > 0:
            for id in dev_ids:
                scene_name = f'{drama_id}_{id}.txt'
                copy_path = os.path.join(folder, scene_name)
                paste_path = os.path.join(save_path, 'dev', scene_name)
                shutil.copy(copy_path, paste_path)
        if len(test_ids) > 0:
            for id in test_ids:
                scene_name = f'{drama_id}_{id}.txt'
                copy_path = os.path.join(folder, scene_name)
                paste_path = os.path.join(save_path, 'test', scene_name)
                shutil.copy(copy_path, paste_path)
    print('data has been split')


def split_by_ratio(l, ratio):
    # 1. make sure train set contains at least one instance
    # 2. If 1 is fulfilled, make sure test/dev sets contain at least one instance (first test, then dev)
    # 3. If 1/2 are fulfilled, round down test/dev set numbers
    assert sum(ratio) == 1
    num = len(l)
    train_num, dev_num, test_num = num * ratio[0], num * ratio[1], num * ratio[2]
    assert num != 0
    if num == 1:
        train_num, dev_num, test_num = 1, 0, 0
    elif num == 2:
        train_num, dev_num, test_num = 1, 0, 1
    elif num == 3:
        train_num, dev_num, test_num = 1, 1, 1
    elif min(train_num, dev_num, test_num) < 1:
        assert min(train_num, dev_num, test_num) != train_num
        if dev_num < 1:
            dev_num = 1
        if test_num < 1:
            test_num = 1
        dev_num = int(dev_num)
        test_num = int(test_num)
        train_num = num - dev_num - test_num
    else:
        dev_num = int(dev_num)
        test_num = int(test_num)
        train_num = num - dev_num - test_num
    return l[:train_num], l[train_num:train_num + dev_num], l[train_num + dev_num:train_num + dev_num + test_num]

if __name__ == '__main__':    
    # parse dracor corpus by scene + save files in  output/dracor (WITH SPEAKERS)
    # save_parsed_data('data/parsed_dracor_speech_by_scenes', prefix='dracor_data', keep_speakers=True, corpus='dracor', separate_by_sentence=True)

    # parse dracor corpus by scene + save files in  output/dracor (WITHOUT SPEAKERS)
    # save_parsed_data('../test_folder/parsed_dracor_speech_by_scenes', prefix='../output/dracor/wo_speakers', keep_speakers=False, corpus='dracor', separate_by_sentence=True)

    #texts, ix = parse_corp_tokenized('../test_folder/parsed_dracor_speech_by_scenes', keep_speakers=True, corpus='dracor')
    #texts, ix = parse_corp_tokenized('../test_folder/parsed_dta_with_scene', keep_speakers=True, corpus='dta', speaker_mappings=speakers)

    # fix speaker info in dta
    # fix_speakers_dta('../pre_processing/dta_processing/data.json', '../pre_processing/dta_processing/fixed_speakers.json')
    # speakers = get_dta_speakers('../pre_processing/dta_processing/fixed_speakers.json')

    # split training data
    # save_parsed_data('../test_folder/parsed_dracor_speech_by_scenes', prefix='../test_folder/dracor_by_scenes', keep_speakers=True, corpus='dracor', separate_by_sentence=True, end_of_text='<endoftext>')
    # split_data('../output/outline/w_speakers/dracor', '../train_data/mod1/dracor_data/outlines', [0.8, 0.1, 0.1])

    # split_data('dracor_data', 'data_split', [0.8, 0.1, 0.1])
    import argparse
    parser = argparse.ArgumentParser(description='parse and split dracor/dta data.')
    parser.add_argument('corpus', choices=['dracor', 'dta'])
    parser.add_argument('in_file', type=str,
                        help='Path to the corpus with plays.')
    parser.add_argument("out_dir", type=str, help="Directory to write parsed data")
    parser.add_argument("split_dir", type=str, help="Directory to write split data for training/dev/test")

    args = parser.parse_args()
    save_parsed_data(args.in_file, prefix=args.out_dir, keep_speakers=True, corpus=args.corpus, separate_by_sentence=True)
    split_data(args.out_dir, args.split_dir, [0.8, 0.1, 0.1])