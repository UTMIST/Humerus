import pathlib

import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class HumerusGPT2:

    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

    def get_humour_score(self, sentence):
        x = self.tokenizer.encode(sentence, return_tensors='pt')
        logits = self.model(x, labels=x)[1]

        x = x.squeeze(0)
        logits = logits.squeeze(0)
        nt_probs = nn.Softmax(dim=1)(logits)

        prob = 0.0
        for i in range(x.size(0)):
            prob += nt_probs[i][x[i].item()].item()

        avg_prob = prob / x.size(0)
        return avg_prob


if __name__ == '__main__':
    # debugging code
    model_dir = pathlib.Path(__file__).parents[1] / 'models'
    model = HumerusGPT2(model_dir)
