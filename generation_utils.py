import transformers

import nltk
import editdistance

SPECIAL_TOKENS = {"bos_token": "<BOS>",
                  "eos_token": "<EOS>",
                  "unk_token": "<UNK>",
                  "pad_token": "<PAD>",
                  "sep_token": "<SEP>"}


def step(model,
         tokenizer,
         input_ids,
         max_length,
         early_stopping,
         no_repeat_ngram_size,
         temperature,
         top_p,
         repetition_penalty,
         top_k,
         do_sample,
         num_beams,
         num_return_sequences,
         characters,
         start_from,
         ):
    outputs = model.generate(input_ids=input_ids,
                             max_length=max_length,
                             num_beams=num_beams,
                             no_repeat_ngram_size=no_repeat_ngram_size,
                             early_stopping=early_stopping,
                             do_sample=do_sample,
                             top_k=top_k,
                             top_p=top_p,
                             temperature=temperature,
                             repetition_penalty=repetition_penalty,
                             num_return_sequences=num_return_sequences)
    if num_return_sequences == 1:
        return drama_decode(outputs[0][start_from:], tokenizer, characters)
    elif num_return_sequences > 1:
        output_sequences = []
        for output in outputs:
            output_sequences.append(drama_decode(output[start_from:], tokenizer, characters))
        return output_sequences


def drama_decode(model_output, tokenizer, characters):
    output_str = tokenizer.decode(model_output, skip_special_tokens=True)
    output_lines_by_n = output_str.split('\n')[:-1]

    output_lines = []
    for line in output_lines_by_n:
        output_sentences = nltk.sent_tokenize(line)
        output_lines += output_sentences

    # remove repeated sentences
    unique_lines = []
    for line in output_lines:
        if line not in unique_lines:
            unique_lines.append(line)
        elif line in characters:
            unique_lines.append(line)

    # remove special tokens
    output_sequences = []
    for line in unique_lines:
        for tok in SPECIAL_TOKENS.values():
            line = line.replace(tok, '')
        output_sequences.append(line)

    # replace name!?,. with name:
    output_sequences = refine_punctuation(characters, output_sequences)

    # remove empty speech
    output_sequences = remove_empty_speeches(output_sequences)

    # refine character names
    # if the edit distance between wrong name and real name is small, replace; else remove
    output_sequences = replace_bad_names(output_sequences, characters)

    # split speaker and speech with line break

    output_sequences = add_line_break(characters, output_sequences)

    # merge speech from the same speaker

    output_sequences = merge_same_speakers(characters, output_sequences)

    return output_sequences


def remove_empty_speeches(lines):
    valid_lines = []
    total_length = len(lines)
    for i in range(total_length - 1):

        if not only_speakers(lines[i]):
            valid_lines.append(lines[i])
        elif contain_speakers(lines[i + 1]) or only_speakers(lines[i+1]):
            continue
        else:
            valid_lines.append(lines[i])
    return valid_lines


def only_speakers(line):
    if line.strip().endswith(':') and len(line.strip()) <= 30:
        return True
    return False


def contain_speakers(line):
    div = line.strip().split(':')
    if len(div) > 1 and len(div[0]) <= 30:
        return True
    return False


def replace_name(name, characters):
    assert name not in characters
    for character in characters:
        if editdistance.eval(name, character) <= 2:
            return character
    return None


def replace_bad_names(lines, characters):
    valid_lines = []
    invalid_speech = False
    for line in lines:
        if not invalid_speech:
            if not contain_speakers(line) and not only_speakers(line):
                valid_lines.append(line)
            else:
                # detect names
                name = line.split(':')[0] + ':'
                if name in characters:
                    valid_lines.append(line)
                else:
                    # decide whether to replace or remove
                    new_name = replace_name(name, characters)
                    if new_name is not None:
                        line = new_name + ':'.join(line.split(':')[1:])
                        valid_lines.append(line)
                    else:
                        invalid_speech = True
        else:
            if not contain_speakers(line) and not only_speakers(line):
                continue
            else:
                name = line.split(':')[0] + ':'
                if name in characters:
                    valid_lines.append(line)
                    invalid_speech = False
                else:
                    new_name = replace_name(name, characters)
                    if new_name is not None:
                        line = new_name + ':'.join(line.split(':')[1:])
                        valid_lines.append(line)
                        invalid_speech = False
                    else:
                        invalid_speech = True
    return valid_lines


def refine_punctuation(characters, output_sequences):
    # name!,.? -> name:
    names = [character[:-1] for character in characters]
    for i in range(len(output_sequences)):
        for name in names:
            if output_sequences[i].find(name) == 0 and len(output_sequences[i]) > len(name) and output_sequences[i][len(name)] != ':':
                output_sequences[i] = name + ':' + ' '.join(output_sequences[i].split(' ')[1:])
                break
    return output_sequences


def add_line_break(characters, output_sequences):
    # name:speech ->
    # name:
    # speech
    outputs = []
    for sequence in output_sequences:
        added = 0
        for character in characters:
            if sequence.find(character) == 0 and len(sequence.strip()) != len(character.strip()):
                outputs.append(character)
                outputs.append(sequence[len(character):] if sequence[len(character)] != ' ' else sequence[len(character)+1:])
                added = 1
                break
        if added == 0:
            outputs.append(sequence)
    return outputs


def merge_same_speakers(characters, output_sequences):
    # A: speech1 A: speech2 -> A: speech1 speech2
    outputs = []
    current_speaker = ''
    for sequence in output_sequences:
        added = 0
        for character in characters:
            if sequence.strip() == character and character == current_speaker:
                added = 1
                break
            elif sequence.strip() == character and character != current_speaker:
                current_speaker = character
                outputs.append(sequence)
                added = 1
                break
        if added == 0:
            outputs.append(sequence)
    return outputs
