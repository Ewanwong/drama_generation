import torch

import gpt2_fine_tuning as gpt2_fine_tuning
from utils.char_support import extract_character_names
from data_process.outline_extraction_text_rank import get_speakers
import utils.generation_utils as generation_utils

import utils.summarize as summarize

SPECIAL_TOKENS = {"bos_token": "<BOS>",
                  "eos_token": "<EOS>",
                  "unk_token": "<UNK>",
                  "pad_token": "<PAD>",
                  "sep_token": "<SEP>"}


class Generator:
    """Slave process, handles GPT2, generates stuff on demand."""

    def __init__(self, model, tokenizer, gen_len=100, separator=None, summarize=True, summ_length=250,
                 special_tokens=None):
        super(Generator, self).__init__()
        self.model_name = model
        self.gen_len = gen_len
        self.max_len = 1024
        self.summarize = summarize
        self.separator = separator

        # number of tokens to summarize
        self.summ_length = summ_length
        self.tokenizer = gpt2_fine_tuning.get_tokenier(tokenizer, special_tokens)
        self.model = gpt2_fine_tuning.get_model(self.model_name, self.tokenizer, special_tokens)

        self.model.cuda()
        self.len_special_token = len(self.tokenizer.encode('<BOS>'))
        if self.separator:
            self.sep_encoding = self.tokenizer.encode(self.separator)
        else:
            self.sep_encoding = []

    def gen_lines(self, outline, prompt, characters=None,
                  limit_characters=True):
        """This is where the generation occurs -- generate self.gen_num continuation
        alternatives for the given prompt.

        TODO: this is the future interface to the method, currently not
        implemented

        prompt = input text

        scene_key = prompt_key-cont_key -- e.g. life-aac will generate
        life-aac (and possibly life-aacaaaa...); will determine random seed

        num_lines = how many lines to generate; this is a maximum, fewer lines
        can be genrated, depending on how many fit into the 1024 tokens window
        (for ptompt + output)

        characters = list of character names allowed in generation; empty =
        generate any character names

        returns a list of generated lines; the first line corresponds to the
        inpout scene_key, the further lines corrspond to "...a" continuations
        """
        # logger.info('GENERATOR forbidden lines: {}'.format(forbidden_lines))

        # based on stuff from interactive.py
        # print('GENERATOR: starting {}...'.format(
        #    repr(prompt[:50])))
        context = self.tokenizer.encode(prompt)

        # TODO: 计算outline长度，总结时不总结outline
        encoded_outline = self.tokenizer.encode(outline)

        # TODO: outline后加<sep>

        if self.summarize and len(context) >= (
                self.max_len - self.gen_len - 2 * len(self.sep_encoding) - 3 * self.len_special_token):
            # print("summarize")
            sum_chunk, raw_chunk = context[len(encoded_outline):-self.summ_length], context[-self.summ_length:]
            # add rest of meaningful string that may be split
            sum_trailing = raw_chunk[:raw_chunk.index(203) + 1] if 203 in raw_chunk else []
            raw_chunk = raw_chunk[raw_chunk.index(203) + 1:] if 203 in raw_chunk else raw_chunk

            sum_token_limit = self.max_len - self.gen_len - len(raw_chunk) - len(encoded_outline) - 2 * len(
                self.sep_encoding) - 3 * self.len_special_token

            pre_sum_ids = sum_chunk + sum_trailing
            sum_str = self.tokenizer.decode(pre_sum_ids)  # XXX djurks please check
            # summarize
            summarized = summarize.summarize_prompt(self.tokenizer, sum_str, sum_token_limit, n_lines=5)
            sum_ids = self.tokenizer.encode(summarized)
            # print(f"SUMMARIZER: summarized  {len(pre_sum_ids)} tokens into => \n {summarized}.")
            context = self.tokenizer.encode(
                '<BOS>') + encoded_outline + self.sep_encoding + sum_ids + self.sep_encoding + raw_chunk + self.sep_encoding
        else:
            context = self.tokenizer.encode(
                '<BOS>') + encoded_outline + self.sep_encoding + self.sep_encoding + context[
                                                                                     len(encoded_outline):] + self.sep_encoding

        assert (len(context) < self.max_len - self.gen_len), "Prompt length out of bound"

        context = context[- self.max_len + self.gen_len:]
        start_from = len(context)
        max_length = start_from + self.gen_len
        context = torch.tensor([context])

        if torch.cuda.device_count() >= 1:
            context = context.to('cuda')

        if limit_characters:
            if characters is None:
                characters = extract_character_names(prompt)

        # returns a list of lines starting with line scene_key
        output_lines = generation_utils.step(
            self.model,
            tokenizer=self.tokenizer,
            input_ids=context,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=4,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.01,
            top_k=50,
            do_sample=True,
            num_beams=1,
            num_return_sequences=10,
            characters=characters,
            start_from=start_from
        )

        # for line in output_sequences:
        # print('GENERATOR truncated: {}...'.format(
        #    repr(line[:50])))

        return {'lines': output_lines}

    def run(self, outline, start, max_iter=10):
        # initialize GPT2 (must be done within the slave process)
        self.max_len = 1024
        print("GENERATOR: Model loaded.")
        # handle requests for generation
        prompt = outline + start
        generated = start
        generated_list = start.split('\n')
        characters = get_speakers(prompt, 30)
        iter = 0

        patience = 0
        while True:
            iter += 1

            results = self.gen_lines(outline, prompt, characters)
            forbidden_lines = generated.split('\n')
            # remove forbidden lines (lines that are already generated)
            for result in results['lines']:
                lines = []
                for line in result:
                    if line not in forbidden_lines:
                        lines.append(line)
                    elif line in characters:
                        lines.append(line)
                    else:
                        continue
                output = '\n'.join(lines) + '\n'
                # if only repeated lines are generated, skip to next return
                if output != '\n':
                    # no speaker in generated lines, skip this return
                    """
                    if no_speaker(lines):
                        patience += 1
                        continue
                    """
                    patience = 0
                    # prevent empty speech
                    if generation_utils.only_speakers(generated_list[-1]) and generation_utils.only_speakers(lines[0]):
                        generated_list = generated_list[:-1]
                        generated_list += lines
                        generated = '\n'.join(generated_list) + '\n'
                    else:
                        generated += output
                        generated_list += lines
                    # print(f"**********************************{generated}******************************")
                    break
                else:
                    patience += 1
            # generate 100 words properly
            if patience == 0:
                prompt = outline + generated  # prompt should contain no special tokens, pure text format
            else:  # if all returns contain only repeated lines, try generation without outline
                output_without_outline, generated_list_without_outline = generate_without_outline(self.model,
                                                                                                  self.tokenizer,
                                                                                                  generated,
                                                                                                  forbidden_lines,
                                                                                                  characters)

                if output_without_outline is not None:
                    # no speakers in the generation, stop generation
                    """
                    if no_speaker(generated_list_without_outline):
                        break
                    """
                    # check empty speech
                    if generation_utils.only_speakers(generated_list[-1]) and generation_utils.only_speakers(
                            generated_list_without_outline[0]):
                        generated_list = generated_list[:-1]
                        generated_list += generated_list_without_outline
                        generated = '\n'.join(generated_list) + '\n'
                    else:
                        generated += output_without_outline
                        generated_list += generated_list_without_outline
                else:  # if generation without outline also fails, stop generation
                    print("generation stops, because no return generates valid output")
                    if generation_utils.only_speakers(generated_list[-1]):  # no ending with only speakers
                        generated_list = generated_list[:-1]
                    break

            # stop conditions: <endoftext> or exceed iteration limits
            if '<endoftext>' in prompt:
                print("end with EOT")
                break
            elif iter > max_iter:
                print("end with maximum iter")
                break

        if '<endoftext>' in generated:
            generated_list = generated.split('<endoftext>')[:-1]
            if generation_utils.only_speakers(generated_list[-1]):
                generated_list = generated_list[:-1]
                # merge speeches from same speakers
                generated_list = generation_utils.merge_same_speakers(characters, generated_list)
            return '\n'.join(generated_list)

        if generation_utils.only_speakers(generated_list[-1]):
            generated_list = generated_list[:-1]
            # merge speeches from same speakers
            generated_list = generation_utils.merge_same_speakers(characters, generated_list)
        return '\n'.join(generated_list)


def generate_without_outline(model, tokenizer, generated, forbidden_lines, characters):
    print("try generating without outline")
    context = tokenizer.encode(generated)[:924]  # <SEP>?
    start_from = len(context)
    context = torch.tensor([context])
    context_len = context.shape[1]
    if torch.cuda.device_count() >= 1:
        context = context.to('cuda')
    output_lines = generation_utils.step(
        model=model,
        tokenizer=tokenizer,
        input_ids=context,
        max_length=context_len + 100,
        early_stopping=True,
        no_repeat_ngram_size=4,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.1,
        top_k=30,
        do_sample=True,
        num_beams=10,
        num_return_sequences=1,
        characters=characters,
        start_from=start_from
    )
    lines = []
    for line in output_lines:
        if line not in forbidden_lines:
            lines.append(line)
        elif line in characters:
            lines.append(line)
        else:
            # print("repeating lines from last step")
            continue
    output = '\n'.join(lines) + '\n'
    if output == '\n':
        return None, None
    else:
        return output, lines


def no_speaker(lines):
    for line in lines:
        if generation_utils.contain_speakers(line) or generation_utils.only_speakers(line):
            return False
    return True
