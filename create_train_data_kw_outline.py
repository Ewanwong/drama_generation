import os
import argparse
import sys
import pickle
from parse_data import parse_corp_tokenized
from keyword_extraction import perform_extr, write_outline
from outline_extraction_text_rank import outline_by_textrank
from utils import save_pkl
from parse_data import split_data

def get_filenames(dir_path:str, extension='.txt'):
    fis = []
    for file in os.listdir(dir_path):
        if file.endswith(extension):
            fis.append(os.path.join(dir_path, file))
    return fis

def load_scenes_wo_speakers(in_files:list):
    scenes = []
    ix = []
    for fi in in_files:
        id = fi[fi.rfind('/') + 1:fi.rfind('.txt')]
        with open(fi, 'r', encoding='utf8') as f:
            cont = [line for line in f if line.strip()]
            scenes.append(''.join(cont))
            ix.append(id)
    return scenes, ix
def load_outlines(in_files:list):
    outlines = []
    ix = []
    for fi in in_files:
        id = fi[fi.rfind('/')+1:fi.rfind('.txt')]
        print(id)
        with open(fi, 'r', encoding='utf8') as f:
            cont = [line for line in f if line.strip()]
            outlines.append(''.join(cont))
            ix.append(id)
    return outlines, ix

def load_keywords(keyw_files:list):
    keyw = []
    ix = []
    for fi in keyw_files:
        id = fi[fi.rfind('/') + 1:fi.rfind('.txt')]
        with open(fi, 'r', encoding='utf8') as f:
            # TODO: double check if splitting is done correctly
            cont = [line.split('\t')[0] for line in f if line.strip()]  # FIXME: in some cases keyw, prob are reversed
            keyw.append(', '.join(cont))    # FIXME: concatenation character?
            ix.append(id)
    return keyw, ix

def load_outlines_dict(in_files:list):
    outlines = dict()
    for fi in in_files:
        id = fi[fi.rfind('/')+1:fi.rfind('.txt')]
        #print(id)
        with open(fi, 'r', encoding='utf8') as f:
            cont = [line for line in f if line.strip()]
            outlines[id] = ''.join(cont)
    return outlines

def load_keywords_dict(keyw_files:list):
    keyw = dict()
    for fi in keyw_files:
        id = fi[fi.rfind('/') + 1:fi.rfind('.txt')]
        with open(fi, 'r', encoding='utf8') as f:
            # TODO: double check if splitting is done correctly
            cont = [line.split('\t')[0] for line in f if line.strip()]  # FIXME: in some cases keyw, prob are reversed
            keyw[id] = ', '.join(cont)    # FIXME: concatenation character?
    return keyw

def generate_keywords(path_in, corpus:str, alg:str, top_k=10, keywords_from_outline=True, num_texts=2):
    print("in generate keywords")
    kw_path_out = 'output/'

    if keywords_from_outline:
        files_in = sorted(get_filenames(path_in))
        # load outlines with speakers
        txts, ids = load_outlines(files_in)

        # extract keywords from outline
        kw_path_out += 'kw/from_outline/{}/{}/'.format(corpus, top_k)

    else:
        # load scenes
        files_scenes = sorted(get_filenames(path_in))
        txts, ids = load_scenes_wo_speakers(files_scenes)

        # extract keywords directly from the scenes + write to file
        kw_path_out += 'kw/from_text/{}/{}/'.format(corpus, top_k)

    if not os.path.exists(kw_path_out):
        os.makedirs(kw_path_out)
    print(files_in)
    if num_texts is None:
        num_texts = len(txts)
    print("num texts: ", num_texts)
    print('Out path: ', kw_path_out, '\tkw from outlines:', keywords_from_outline, '\tAlg: ', alg, '\t top_k: ', top_k)
    perform_extr(kw_path_out, txts[0:num_texts], ids, alg, top_k=top_k, print_to_screen=False)



def generate_outlines(file_path:str, corpus:str, num_texts=2):

    # load the corpus
    txts, ids = parse_corp_tokenized(file_path, keep_speakers=True,
                                     corpus=corpus, add_newline=True)
    outline_out = 'output/outline/{}/{}/'.format('w_speakers', corpus)
    outline_out_wo_speakers = 'output/outline/{}/{}/'.format('wo_speakers', corpus)
    if num_texts is None:
        num_texts = len(txts)

    if not os.path.exists(outline_out_wo_speakers):
        os.makedirs(outline_out_wo_speakers)
    if not os.path.exists(outline_out):
        os.makedirs(outline_out)
    # extract outlines with and without speakers, save
    for i, x in enumerate(txts[0:num_texts]):
        outl, outl_wo_speakers = outline_by_textrank(x)
        write_outline(outline_out + str(ids[i]) + '.txt', outl)
        write_outline(outline_out_wo_speakers + str(ids[i]) + '.txt', outl_wo_speakers)
        #print('Outline:', outl)
        #print('Outline:', outl_wo_speakers)
        print(i)



def create_train_pkl(outline_dir:str, keywords_dir:str, out_dir:str, alg:str, k:int, type='outl'):
    if outline_dir[-1] != '/':
        outline_dir += '/'
    if keywords_dir[-1] != '/':
        keywords_dir += '/'
    if out_dir[-1] != '/':
        out_dir += '/'

    for x in ['test', 'dev', 'train']:
        out_dict = {}
        outl_dir = outline_dir + x
        key_dir = keywords_dir + x
        # get outline file names
        outline_files = get_filenames(outl_dir)
        # load outlines
        outlines = load_outlines_dict(outline_files)

        # load keywords
        keyword_files = get_filenames(key_dir)
        keywords = load_keywords_dict(keyword_files)

        # combine into a dictionary
        for key in outlines.keys():
            out_dict[key] = [keywords[key], outlines[key]]
            if key not in keywords.keys():
                print(key, 'is not in keyword ids')
                break
            print(key, keywords[key], outlines[key])
        for key in keywords.keys():
            if key not in outlines.keys():
                print(key, 'is not in outline ids')
                break

        # save as pkl
        output_dir = out_dir + alg + '/' + type + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_file = output_dir + f'{x}' + '_' + alg + '_' + type + '_' + str(k) + '.pkl'

        save_pkl(out_file, out_dict)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare training data for model 1. '
                                                 'Extracts keywords or outlines from scenes or prepare a pkl file with'
                                                 'training data (keywords-outlines) for training GPT-2.')
    parser.add_argument('action', choices=['outlines', 'keywords_text', 'keywords_outlines', 'train_data', 'split'],
                        help='outlines: extract outlines with and without speakers,'
                             'keywords_text: extract keywords directly from each scene,'
                             'keywords_outlines: extract keywords from outlines of each scene,'
                              'split: if the data should be split into train, dev, test sets first.'
                             'train_data: create training data from keyword and outline files')
    parser.add_argument('in_file', type=str,
                        help='Path to the corpus with plays or to directory with outlines. If keywords_text, then path to corpus file'
                             'If keywords_outlines then path to dir with outlines')
    parser.add_argument('corpus_name', type=str,
                        help='dracor or dta')
    parser.add_argument('--alg', type=str,
                        help="'tfidf', 'textrank', 'rake', 'yake', 'keybert', 'multirake',  'textrank'")
    parser.add_argument('--k', type=int,
                        help='Number of keywords to extract')
    parser.add_argument('--train_out', type=str,
                        help='name of output file to which the training set should be written.')
    parser.add_argument('--keywords_dir', type=str,
                        help='path to the directory with files containing the directory with keywords train/dev/test split')
    parser.add_argument('--data_dir', type=str,
                        help='path to the directory with files containing keywords or outlines for each text/scene')
    parser.add_argument('--outline_dir', type=str,
                        help='path to the directory with files containing outlines for each text/scene')
    parser.add_argument('--num_texts', type=int,
                        help='Number of texts from the corpus that should be processed')
    parser.add_argument('--kw_type', type=str,
                        help='Only used with train_data, values = [outl, scene]')
    parser.add_argument('--out_dir', type=str,
                        help='Output directory to which the train/dev/test split will be saved')
    args = parser.parse_args()

    print()
    in_path = args.in_file      # '../test_folder/parsed_dracor_no_names_full'
    corpus = args.corpus_name   # 'dracor'
    alg = args.alg              # 'textrank'
    k = args.k                  # 20
    txts_to_process = args.num_texts

    #print(os.getcwd())
    print(args)
    if args.action == 'outlines':
        print("outlines")
        # python3 pipeline/create_train_data_kw_outline.py outlines test_folder/parsed_dracor_speech_by_scenes dracor --num_texts 2
        generate_outlines(in_path, corpus=corpus, num_texts=txts_to_process)

    elif args.action == 'keywords_text' or args.action == 'keywords_outlines':
        print("keywords")
        # From outlines: python3 pipeline/create_train_data_kw_outline.py keywords_outlines output/outline/wo_speakers/dracor dracor --alg textrank --k 10
        # From scenes: python3 pipeline/create_train_data_kw_outline.py keywords_text output/dracor/wo_speakers dracor --alg textrank --k 20

        from_outline = False
        if alg is None:
            sys.stderr.write('Please pass the --alg argument (algorithm for keyword extraction)')
            sys.exit(1)
        if k is None:
            sys.stderr.write('Please pass the --k argument (number of keywords to extract)')
            sys.exit(1)
        if args.action == 'keywords_outlines':
            from_outline = True
        print('start')
        generate_keywords(in_path, corpus, alg, top_k=k, keywords_from_outline=from_outline,
                          num_texts=txts_to_process)
    elif args.action == 'split':
        # split keywords:
        # python3 pipeline/create_train_data_kw_outline.py split bla.txt dracor --data_dir output/kw/from_outline/dracor/10/tfidf --out_dir train_data/mod1/dracor_data/kw_outline/10/tfidf

        # split outlines (only once!)
        # python3 pipeline/create_train_data_kw_outline.py split bla.txt dracor --data_dir output/outline/w_speakers/dracor --out_dir train_data/mod1/dracor_data/outlines/
        if args.data_dir is None:
            sys.stderr.write('Please pass the --data_dir argument (dir with keyword or outline files that should be split)')
            sys.exit(1)

        if args.out_dir is None:
            sys.stderr.write('Please pass the --out_dir argument (Output directory to which the train/dev/test split will be saved)')
            sys.exit(1)

        split_data(args.data_dir, args.out_dir,[0.8, 0.1, 0.1])
    elif args.action == 'train_data':
        # from outlines with speakers + keywords from outlines (extracted via textrank + 10 keywords per outline)
        # python3 pipeline/create_train_data_kw_outline.py train_data bla.txt dracor --train_out train_data/mod1/split/ --keywords_dir train_data/mod1/dracor_data/kw_outline/10/textrank  --outline_dir train_data/mod1/dracor_data/outlines --k 10 --alg textrank --kw_type outl

        # from outlines with speakers + keywords from scenes (extracted via textrank + 20 keywords per outline)
        # python3 pipeline/create_train_data_kw_outline.py train_data bla.txt dracor --train_out train_data/mod1/split/ --keywords_dir train_data/mod1/dracor_data/kw_text/20/textrank  --outline_dir train_data/mod1/dracor_data/outlines --k 20 --alg textrank --kw_type text

        if args.train_out is None:
            sys.stderr.write('Please pass the --train_out argument (output .pkl file to which the training set will be written)')
            sys.exit(1)
        if args.keywords_dir is None:
            sys.stderr.write('Please pass the --keywords_dir argument (directory with keyword files)')
            sys.exit(1)
        if args.outline_dir is None:
            sys.stderr.write('Please pass the --outline_dir argument (directory with outline files)')
            sys.exit(1)
        if alg is None:
            sys.stderr.write('Please pass the --alg argument (algorithm for keyword extraction)')
            sys.exit(1)
        if k is None:
            sys.stderr.write('Please pass the --k argument (number of keywords to extract)')
            sys.exit(1)
        if args.kw_type is None:
            sys.stderr.write('Please pass the --kw_type argument (what were the keywords extracted from: outline or scenes/text?)')
            sys.exit(1)
        print(args.outline_dir, args.keywords_dir, args.train_out)

        create_train_pkl(args.outline_dir, args.keywords_dir, args.train_out, alg, k, type = args.kw_type)
    else:
        print("else")
        # test load_outlines
        tmp = get_filenames('output/outline/w_speakers/dracor')
        o, i = load_outlines(tmp)
        for x, y in zip(o,i):
            print(y + ':')
            print(x)
            print()