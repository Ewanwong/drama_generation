from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
import torch
import random


def run_eval(file):
    random.seed(1)
    device = "cuda"
    model_id = "dbmdz/german-gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    ppls = []
    max_length = 1024
    stride = 100
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        print(len(text.split("-------------")))
        for t in text.split("-------------"):

            encodings = tokenizer("\n".join(t), return_tensors="pt")

            seq_len = encodings.input_ids.size(1)

            nlls = []
            prev_end_loc = 0
            for begin_loc in tqdm(range(0, seq_len, stride)):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)

                    # loss is calculated using CrossEntropyLoss which averages over input tokens.
                    # Multiply it with trg_len to get the summation instead of average.
                    # We will take average over all the tokens to get the true average
                    # in the last step of this example.
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
            if len(nlls) > 0:
                ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
                print(ppl)
                ppls.append(ppl)

    return sum(ppls) / len(ppls)


avg_ppl = run_eval("not_fine_tuned.txt")
print(avg_ppl)
