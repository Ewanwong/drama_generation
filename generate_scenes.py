import utils.generator as generator
import os
import pickle
import random
import argparse
random.seed(1)
SPECIAL_TOKENS = {"bos_token": "<BOS>",
                  "eos_token": "<EOS>",
                  "unk_token": "<UNK>",
                  "pad_token": "<PAD>",
                  "sep_token": "<SEP>"}


def evaluate_model(model_path, tokenizer_path, test_folder, outline_folder, use_outline=True, n_scenes=5,
                   write_path='./examples.txt'):
    generator = generator.Generator(model_path, tokenizer_path)

    with open(write_path, 'w', encoding='utf-8') as g:
        test1 = os.listdir(outline_folder)
        test2 = os.listdir(test_folder)
        paths = [path for path in test1 if path in test2][:n_scenes]
        for path in paths:
            g.write("---------------------------------------------" + '\n')
            g.write(str(path) + '\n')

            print(path)

            if use_outline:
                outline_path = os.path.join(outline_folder, path)
                with open(outline_path, 'r', encoding='utf-8') as f:
                    outline = f.read()
            else:
                outline = ''
            with open('test_data.pkl', 'rb') as f:
                d = pickle.load(f)
            if len(d[path])>1:
                input = d[path][1][0]
            else:
                continue
            prompt = input.split('<SEP>')[-1]
            print(outline)
            g.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" + '\n')
            g.write("OUTLINE:" + '\n')
            g.write(outline)
            g.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" + '\n')
            print(prompt)
            g.write("START_OF_SCENE:" + '\n')
            g.write(prompt)
            g.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" + '\n')
            generated = generator.run(outline, prompt, max_iter=10)
            print("-----------------------------------------")
            print(generated)
            print("-----------------------------------------")
            g.write("GENERATED_SCENE:" + '\n')
            g.write(generated)


def get_gold_scenes(test_folder, n_scenes=5, write_path='./gold_passage.txt'):
    with open(write_path, 'w', encoding='utf-8') as g:
        paths = os.listdir(test_folder)[:n_scenes]
        for path in paths:
            g.write("---------------------------------------------" + '\n')
            g.write(str(path) + '\n')

            print(path)
            scene_path = os.path.join(test_folder, path)
            with open(scene_path, 'r', encoding='utf-8') as f:
                passage = f.read()
            g.write(passage)


if __name__ == '__main__':
    # evaluate_model('no_outline_model', 'dbmdz/german-gpt2', '../output/dracor_data/test/', '../output/outline/w_speakers/dracor', use_outline=False, write_path='no_outline_output.txt')
    # get_gold_scenes('../output/dracor_data/test/', n_scenes=100, write_path='./100_gold_scenes.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("test_folder", type=str)
    parser.add_argument("--tokenizer_path", type=str, default='dbmdz/german-gpt2')
    parser.add_argument("--outline_folder", type=str, default=None, help="set to None when --use_outline=False")
    parser.add_argument("--use_outline", type=bool, default=False)
    parser.add_argument("--n_scenes", type=int, default=5)
    parser.add_argument("--write_path", type=str)
    args = parser.parse_args()
    evaluate_model(args.model_path, args.tokenizer_path, args.test_folder, args.outline_folder, args.use_outline, args.n_scenes, args.write_path)


