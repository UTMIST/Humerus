import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def log_prob_sentence(sentence):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True)
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    m = torch.nn.Softmax(dim=1)
    output = m(logits)
    log_prob = 0
    for i in range(list(output.size())[1]):
        log_prob = log_prob + np.log(float(output[0][i][0]))

    return log_prob



if __name__ == '__main__':
    print(log_prob_sentence("Fuck Praxis"))