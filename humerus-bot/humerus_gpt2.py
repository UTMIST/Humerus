import pathlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class HumerusGPT2:

    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

    def get_humour_score(self, sentence):
        x = self.tokenizer.encode(sentence, return_tensors='pt')
        x = x.unsqueeze(0)

        logits = self.model(x)[1]
        print(logits)


model_dir = pathlib.Path(__file__).parents[1] / 'models'
model = HumerusGPT2(model_dir)
model.get_humour_score("Once upon a time")
