import sys
import os
import gpt2_fine_tuning as gpt2_fine_tuning
from data_process.keyword_extraction import apply_keyw_extraction
from data_process.create_train_data_kw_outline import get_filenames

SPECIAL_TOKENS = {"bos_token": "<BOS>",
                  "eos_token": "<EOS>",
                  "unk_token": "<UNK>",
                  "pad_token": "<PAD>",
                  "sep_token": "<SEP>"}


def load_keywords(dir_path: str):
    files_in = sorted(get_filenames(dir_path))
    keyws = []
    ids = []
    for f in files_in:
        filename = f[f.rfind('/') + 1:]
        with open(f, 'r', encoding='utf8') as fi:
            keyws.append([l.strip().split('\t')[0] for l in fi])
            ids.append(filename)
    return keyws, ids


def write_generated_outl_to_file(outlines: list, file_ids: list, out_path: str):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not out_path.endswith('/'):
        out_path += '/'

    for outl, id in zip(outlines, file_ids):
        outfile = out_path + id
        with open(outfile, 'w', encoding='utf8') as f:
            # print(outfile)
            #print(type(outl), outl)
            f.write(outl)


class OutlineGenerator:

    def __init__(self, model, tokenizer, separator=None, special_tokens=SPECIAL_TOKENS, max_len=1024):
        super(OutlineGenerator, self).__init__()
        self.model_name = model
        self.max_len = max_len
        # load tokenizer and model
        self.tokenizer = gpt2_fine_tuning.get_tokenier(
            tokenizer, special_tokens)
        self.model = gpt2_fine_tuning.get_model(
            self.model_name, self.tokenizer, special_tokens)
        self.model.cuda()

    def get_keywords(self, text: str, algorithm: str, num_keyw=10):
        """
        Extracts keywords from a text/scene.
        @param text: a string representing a text/scene
        @param algorithm: keyword extraction algorithm
        @param num_keyw: number of keywords to extract from a text
        @return: a list of keywords representing text
        """
        keywords = apply_keyw_extraction(algorithm, [text], top_k=num_keyw)
        if len(keywords) == 1:
            keywords = keywords[0]
        else:
            print('error in get_keywords')
            sys.exit(1)  # this shouldn't happen
        return [kw[0] for kw in keywords]  # drop score, keep keywords

    def generate_outline(self, keywords: list, keyword_sep=',', num_outlines=10):
        """
        Generates an outline for a text based on a list of keywords that represent the text.

        @param keyword_sep: separator for keywords
        @param keywords: a list of keywords representing a text
        @param num_outlines: num of outlines to generate (for a single text)
        @return: a list of outlines for text
        """

        # prepend with bos/eos, separate by commas
        keywords_input = '<BOS>' + keyword_sep.join(keywords) + '<SEP>'

        # tokenize
        generated = self.tokenizer(
            keywords_input, return_tensors="pt").input_ids.cuda()
        """
        generated = torch.tensor(self.tokenizer.encode(keywords_input)).unsqueeze(0)
        device = torch.device("cuda")
        generated = generated.to(device)
        """
        # generate
        generated_outlines = self.model.generate(  # self.model,
            # tokenizer=self.tokenizer,
            # input_ids=generated,
            generated,
            do_sample=True,
            min_length=50,
            max_length=self.max_len,
            top_k=30,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=2.0,
            num_return_sequences=num_outlines
        )
        # decode
        return [self.tokenizer.decode(outline, skip_special_tokens=True) for outline in generated_outlines]


if __name__ == '__main__':
    num_outlines = 1
    # generate outlines for the test set (from keywords), using 2 different models
    paths = {'./models/outline/kw_outl_textr_10': './train_data/mod1/dracor_data/kw_outline/10/textrank/test',
             './models/outline/kw_outl_tfidf_10': './train_data/mod1/dracor_data/kw_outline/10/tfidf/test',
             './models/outline/kw_outl_textr_10_epochs5_warmup200': './train_data/mod1/dracor_data/kw_outline/10/textrank/test'}
    out_path = '../generated_outlines/'

    # './models/outline/kw_outl_textr_10', './models/outline/kw_outl_tfidf_10', 'kw_outl_textr_10_epochs5_warmup200']:
    for model in ['./models/outline/kw_outl_textr_10_epochs5_warmup200']:
        # 1. load correct model and tokenizer
        out = OutlineGenerator(model, 'dbmdz/german-gpt2')
        out_dir = out_path + model[model.rfind('/')+1:]
        print(out_dir)
        # 2. first extract keywords from text (the text should not contain any speakers!) -> load already extracted keywords
        kws, idxs = load_keywords(paths[model])

        # then, generate outlines from keywords
        gener_outlines = []
        for kw, id in zip(kws, idxs):
            o = out.generate_outline(
                kw,  keyword_sep=',', num_outlines=num_outlines)[0]
            kw_str = ','.join(kw)
            wo_kw = o[o.find(kw_str)+len(kw_str):]
            #print(type(wo_kw), wo_kw)
            #print(kw, '\t', o)
            # returns 1 outline for a single text
            gener_outlines.append(wo_kw)
            #write_generated_outl_to_file([wo_kw], [id], out_path=out_dir)
        print('finished generating with: ', model)
        write_generated_outl_to_file(gener_outlines, idxs, out_path=out_dir)
