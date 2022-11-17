import pickle

ALLOWED_CORPORA = {
    'dracor': {'play_id_start': '$id_', 'play_id_end': '@id_', 'speaker_start': '$sp', 'speaker_end': '@sp_',
               }, 'dta': {'play_id_start': 'DRAMA_NAME: ', 'play_id_end': 'DRAMA_END:', 'speaker_start': '$sp',
                          'speaker_end': '@sp_'}}


def parse_corp(filename: str, keep_speakers=False, corpus='dracor', add_newline=True):
    if corpus not in ALLOWED_CORPORA:
        raise ValueError(f'{corpus} is not in allowed corpora. Please choose from {ALLOWED_CORPORA.keys()}')

    play_start = ALLOWED_CORPORA[corpus]['play_id_start']
    play_end = ALLOWED_CORPORA[corpus]['play_id_end']
    sp_start = ALLOWED_CORPORA[corpus]['speaker_start']
    sp_end = ALLOWED_CORPORA[corpus]['speaker_end']
    texts = []
    ids = []

    with open(filename, 'r', encoding='utf8') as f:
        text = ''
        cur_id = ''
        for line in f:
            line = line.strip()  # TODO: leave \n in the text?
            # start of the play: get the id
            if line.startswith(play_start):
                if corpus == 'dta':
                    cur_id = line.split(play_start)[1]
                else:
                    cur_id = line
            # end of the play: add text to list of play draco_texts
            elif line.startswith(play_end):
                if text == '':
                    print(cur_id, ' is empty!')
                    continue
                texts.append(text)  # TODO: maybe need to add a separator?
                ids.append(cur_id)
                cur_id = ''
                text = ''
            # get speakers if needed
            elif line.startswith(sp_start):  # $sp_#die_diener_1-1
                if keep_speakers:
                    speakers = line.split('#')[1:]
                    if len(speakers) > 1:
                        speaker = ''
                        for i, sp in enumerate(speakers):
                            sp = split_names(sp)
                            if i == len(speakers) - 1:
                                speaker += sp.capitalize()
                            else:
                                speaker += sp.strip().capitalize() + ', '
                    else:
                        speaker = split_names(''.join(speakers))
                    text += ' ' + speaker + ':\n'
            # end of speech
            elif line.startswith(sp_end):
                if add_newline:
                    text += '\n'
                else:
                    text += ' '
            else:
                text += line + ' '

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
    print(speaker)
    return speaker


def load_pkl(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pkl(filename:str, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == '__main__':
    # txs, ix = parse_corp('../outline_extraction/parsed_dta', keep_speakers=False, corpus='dta')
    txs, ix = parse_corp('../test_folder/parsed_dta', keep_speakers=True, corpus='dta')
    # print(ix)
    prefix = '../test_folder/dta_texts/'
    for id, text in zip(ix, txs):
        with open(prefix+id, 'w', encoding='utf-8') as f:
            f.write(text)
