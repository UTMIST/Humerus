import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def log_prob_sentence(sentence):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True)
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs['input_ids'].numpy()
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    m = torch.nn.Softmax(dim=1)
    output = m(logits).detach().numpy()
    log_prob = 0
    for i in range(output.shape[1]):
        log_prob = log_prob + np.log(float(output[0][i][input_ids[0][i]]))

    return log_prob



if __name__ == '__main__':
    print(log_prob_sentence("Fuck Praxis"))