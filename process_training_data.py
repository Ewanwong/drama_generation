import glob
from transformers import AutoTokenizer, AutoModelWithLMHead

import generation_utils
from outline_extraction_text_rank import outline_by_textrank, get_speakers
import os
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import pickle
import summarize
from tqdm import tqdm
import multiprocessing

SPECIAL_TOKENS = {"bos_token": "<BOS>",
                  "eos_token": "<EOS>",
                  "unk_token": "<UNK>",
                  "pad_token": "<PAD>",
                  "sep_token": "<SEP>"}


class TrainingDataGenerator:
    def __init__(self, folder, outlines, model, num_special_tokens=3, separator=None, max_len=1024, gen_len=100,
                 sum_length=250,
                 outline_length=150,
                 name_length=20, language='german'):
        self.tokenizer, self.model = generation_utils.setup(model, SPECIAL_TOKENS)
        self.num_special_tokens = num_special_tokens
        self.separator = separator  # outline <sep> summary <sep> local_context
        self.max_len = max_len  # length limit of the language model
        self.gen_len = gen_len  # maximum number of words generated at each step
        self.name_length = name_length  # maximum length of character name
        self.outline_length = outline_length  # maximum length of outline
        self.summ_length = sum_length  # number of previous words that will not be reduced
        self.language = language
        self.folder = folder
        self.outline_folder = outlines
        self.training_pairs = defaultdict(list)
        self.len_special_token = len(self.tokenizer.encode('<BOS>'))
        if self.separator:
            self.sep_encoding = self.tokenizer.encode(self.separator)
        else:
            self.sep_encoding = []

    def _get_outline(self, scene) -> str:
        # scene: file name
        # outline: str
        outline = outline_by_textrank(scene, words=self.outline_length, name_length=self.name_length,
                                      language=self.language)
        return outline

    def _read_outline(self, name) -> str:
        outline_path = os.path.join(self.outline_folder, name)
        with open(outline_path, 'r', encoding='utf-8') as f:
            outline = f.read()
        return outline

    def _generate_training_data(self, scene, name):
        # name: file name
        # outline = self._get_outline(scene)  # TODO: load from files by name
        outline = self._read_outline(name)
        sents = scene.split('\n')
        sents = [sent + '\n' for sent in sents]

        prompt_upper_length = self.max_len - self.gen_len - self.len_special_token * self.num_special_tokens

        prompt = outline
        sent_id = 0
        encoded_outline = self.tokenizer.encode(outline)
        len_outline = len(encoded_outline)  # length of outline (after tokenization)

        training_pairs = []

        while sent_id <= len(sents) - 1:  # stop when all sentences are generated
            len_generated = 0
            text_generated = ''
            # prompt都为outline <sep> <sep> generated
            context = self.tokenizer.encode(prompt)
            if len(context) > prompt_upper_length - 2 * len(self.sep_encoding):
                # print("summarize")
                # TODO: 要排除<sep>找到raw_text
                sum_chunk, raw_chunk = context[len(encoded_outline):-self.summ_length], context[-self.summ_length:]
                # add rest of meaningful string that may be split
                sum_trailing = raw_chunk[:raw_chunk.index(203) + 1] if 203 in raw_chunk else []
                raw_chunk = raw_chunk[raw_chunk.index(203) + 1:] if 203 in raw_chunk else raw_chunk

                # TODO：此处要再减去两个sep的长度
                if self.separator:
                    sum_token_limit = self.max_len - self.gen_len - len(raw_chunk) - len(
                        encoded_outline) - self.len_special_token * self.num_special_tokens - 2 * self.len_special_token
                else:
                    sum_token_limit = self.max_len - self.gen_len - len(raw_chunk) - len(
                        encoded_outline) - self.len_special_token * self.num_special_tokens

                pre_sum_ids = sum_chunk + sum_trailing
                sum_str = self.tokenizer.decode(pre_sum_ids)  # XXX djurks please check
                # summarize
                summarized = summarize.summarize_prompt(self.tokenizer, sum_str, sum_token_limit, n_lines=5)
                if not summarized:
                    return name, training_pairs
                sum_ids = self.tokenizer.encode(summarized)
                # print(f"SUMMARIZER: summarized  {len(pre_sum_ids)} tokens into => \n {summarized}.")

                # TODO: 改变<sep>位置
                if self.separator:
                    context = context[
                              :len(encoded_outline)] + self.sep_encoding + sum_ids + self.sep_encoding + raw_chunk
                else:
                    context = encoded_outline + sum_ids + raw_chunk
            else:
                if self.separator:
                    context = context[:len(encoded_outline)] + self.sep_encoding + self.sep_encoding + context[
                                                                                                       len(encoded_outline):]
            """
            if len(context) > prompt_upper_length:
                print(len(self.sep_encoding))
                print(len(encoded_outline))
                print(sum_ids)
                print(len(raw_chunk))
            """
            assert (len(context) <= prompt_upper_length), "Prompt length out of bound" + str(len(context)) + '>' + str(
                prompt_upper_length)
            # context = context[- self.max_len + self.gen_len:]
            prompt = self.tokenizer.decode(context)

            while True:  # stop when enough words are generated or reach the end of text
                if sent_id == len(sents):
                    break
                next_sent = sents[sent_id]

                # 长到无法总结
                if len_generated == 0 and len(self.tokenizer.encode(next_sent)) > self.summ_length:
                    print(f"{name} has been read.")
                    return name, training_pairs

                # 一个句子就比100长，略过该句
                if len_generated == 0 and len(self.tokenizer.encode(next_sent)) > self.gen_len:
                    sent_id += 1
                    prompt = outline + '\n' + ''.join(sents[:sent_id])
                    break

                if len_generated + len(self.tokenizer.encode(next_sent)) <= self.gen_len:
                    sent_id += 1
                    text_generated += next_sent
                    len_generated += len(self.tokenizer.encode(next_sent))
                else:
                    break
            assert (len(self.tokenizer.encode(text_generated)) <= self.gen_len)
            training_pairs.append([prompt, text_generated])
            # modify prompt for next generation step
            prompt = outline + ''.join(sents[:sent_id])
        print(f"{name} has been read.")
        return name, training_pairs

    def generate_training_data(self):
        paths = glob.glob(os.path.join(self.folder, '*'))
        scenes = []
        names = []

        for path in paths:
            # TODO: 注意不同系统
            name = path.split('/')[-1]
            names.append(name)
            with open(path, 'r', encoding='utf-8') as f:
                scenes.append(f.read())
        """
        i = 1
        for scene, name in zip(scenes, names):
            print(name)
            name, pairs = self._generate_training_data(scene, name)
            print(str(i) + ' plays are processed: '+name)
            i += 1
            self.training_pairs[name] = pairs
        """
        with multiprocessing.Pool(processes=10) as pool:
            res = pool.starmap_async(self._generate_training_data, zip(scenes, names))
            for name, pairs in res.get():
                self.training_pairs[name] = pairs

        print(f"{len(names)} scenes have been read.")

    def save_training_pairs(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.training_pairs, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='process data for training')
    parser.add_argument('data_dir', type=str, help='Path to the corpus with plays.')
    parser.add_argument('outline_dir', type=str, help='Path to directory to write outlines')
    parser.add_argument('output_path')

    args = parser.parse_args()
    data_folder = args.data_dir
    outline_folder = args.outline_dir
    a = TrainingDataGenerator(data_folder, outline_folder, 'dbmdz/german-gpt2', separator='<SEP>')
    a.generate_training_data()
    print("reading data done, start writing into file")
    a.save_training_pairs(args.output_path)
