from torch.utils.data import Dataset, random_split
from torch import tensor
import torch
import random
import generation_utils
import os
import argparse
from utils import *
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoModelWithLMHead
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
    AdamW, get_linear_schedule_with_warmup, \
    TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy

SPECIAL_TOKENS = {"bos_token": "<BOS>",
                  "eos_token": "<EOS>",
                  "unk_token": "<UNK>",
                  "pad_token": "<PAD>",
                  "sep_token": "<SEP>"}
random.seed(1)
MAXLEN = 1024

DEVICE = torch.device("cuda:6" if torch.cuda.is_available() else 'cpu')


class myDataset(Dataset):

    def __init__(self, filename, tokenizer, model, model_type='model2'):
        self.tokenizer = tokenizer
        self.model = model
        # self.model, self.tokenizer = gpt2.setup((model_name, SPECIAL_TOKENS))
        data = load_pkl(filename)
        prompts = []
        generations = []
        samples = []
        for k, v in data.items():
            if model_type == 'model2':
                for x in v:  # x is [prompt, generated_text]
                    # add EOS/BOS tags?
                    prompt = '<BOS>' + x[0]
                    generated = x[1] + '<EOS>'
                    prompts.append(prompt)
                    generations.append(generated)
                    input = prompt + '<SEP>' + generated

                    encoding_dict = self.tokenizer(input,
                                                   truncation=True,
                                                   max_length=MAXLEN,
                                                   padding="max_length")

                    samples.append(encoding_dict)
            else:
                prompt = '<BOS>' + v[0]
                generated = v[1] + '<EOS>'
                prompts.append(prompt)
                generations.append(generated)
                input = prompt + '<SEP>' + generated
                encoding_dict = self.tokenizer(input,
                                               truncation=True,
                                               max_length=MAXLEN,
                                               padding="max_length")

                samples.append(encoding_dict)

        # self.prompts = prompts
        # self.generations = generations
        random.shuffle(samples)
        self.samples = samples

        # ---------------------------------------------#

    # ---------------------------------------------#

    def __len__(self):
        return len(self.samples)

    # ---------------------------------------------#

    def __getitem__(self, i):

        encodings_dict = self.samples[i]

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        """
        return {
            'input_ids': torch.tensor(input_ids).to(DEVICE),
            'attention_mask': torch.tensor(attention_mask).to(DEVICE)}
        """
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)}

    def add_tags(self, sent: str, bos='<BOS>',
                 eos='<EOS>'):  # TODO: change bos and eos? Length limit might need changing
        return bos + '\n' + sent + eos + '\n'

def get_tokenier(tokenizer_path, special_tokens=None, use_auth_token=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=use_auth_token)  # GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer


def get_model(MODEL, tokenizer, special_tokens=None, load_model_path=None):
    # GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else:
        config = AutoConfig.from_pretrained(MODEL,
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)

        # ----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        # Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    # model.cuda()
    return model


class GPT2Training:
    def __init__(self, tokenizer_path: str, model_name: str, train_path: str, dev_path: str, out_dir: str,
                 model_type='model2', early_stopping_patience=10):
        self.tokenizer = get_tokenier(tokenizer_path, SPECIAL_TOKENS)
        self.model = get_model(model_name, self.tokenizer, SPECIAL_TOKENS)
        self.model.eval()
        self.patience = early_stopping_patience
        # self.model.to(DEVICE)

        self.training_args = TrainingArguments(
            output_dir=out_dir,  # "./gpt2-ger-drama",  # The output directory
            overwrite_output_dir=True,  # overwrite the content of the output directory
            num_train_epochs=3,  # number of training epochs
            per_device_train_batch_size=4,  # batch size for training
            per_device_eval_batch_size=8,  # batch size for evaluation
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=400,  # Number of update steps between two evaluations.
            save_steps=800,  # after # steps model is saved
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            load_best_model_at_end=True

        )

        # print(self.training_args.device)
        self.train_set = myDataset(train_path, self.tokenizer, self.model, model_type=model_type)
        print("train set loaded")
        self.dev_set = myDataset(dev_path, self.tokenizer, self.model, model_type=model_type)
        print("dev set loaded")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)  # forms batches
        self.trainer = self.load_trainer()

    def train(self):
        self.trainer.train()
        self.trainer.save_model()

    def load_trainer(self):
        return Trainer(model=self.model, args=self.training_args, data_collator=self.data_collator,
                       train_dataset=self.train_set, eval_dataset=self.dev_set, callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.patience)])  # todo: prediction-only?


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine-tune the GPT-2 German model.')
    parser.add_argument('train', type=str,
                        help='Path to the pkl file with the training data')
    parser.add_argument('dev', type=str,
                        help='Path to the pkl file with the dev data')
    parser.add_argument('out_dir', type=str,
                        help='Path to the output directory')
    parser.add_argument('model_type', type=str,
                        help='Model type: model1 is a model that produces outlines from keywords, '
                             'model2 is the model which generates texts from prompt (outline + summary)')
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    tr = GPT2Training('dbmdz/german-gpt2', 'dbmdz/german-gpt2', args.train, args.dev, args.out_dir,
                      model_type=args.model_type, early_stopping_patience=args.patience)

    tr.train()
