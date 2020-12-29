"""
A script to embed the CAH sentences.
"""

import itertools
import pathlib
import pickle
import time

import json5
import tqdm
import wget
from sentence_transformers import SentenceTransformer

from humerusbot.cardutils import CardUtils

PARENT_DIR = pathlib.Path(__file__).parents[2] / 'datasets'
DECK_DIR = PARENT_DIR / 'deck.json5'
SENTENCE_DIR = PARENT_DIR / 'sentences.txt'
EMBED_DIR = PARENT_DIR / 'embeddings.pkl'


def download_deck():
    if not DECK_DIR.is_file():
        wget.download(url="https://raw.githubusercontent.com/"
                          "UTMIST/humerusdecks/master/"
                          "server/decks/cah-base-en.deck.json5",
                      out=str(DECK_DIR))


def load_raw_cards():
    with open(DECK_DIR, encoding='utf-8') as f:
        raw_data = json5.load(f)

    black_cards = raw_data['calls']
    white_cards = raw_data['responses']

    return black_cards, white_cards


def generate_sentences(black_cards, white_cards, verbose=True,
                       max_slots=1):
    if SENTENCE_DIR.is_file():
        return

    sentences = []
    card_stats = [0, 0, 0, 0]
    for black_card in tqdm.tqdm(black_cards, desc='Processing Black Cards'):
        num_slots = CardUtils.get_num_slots(black_card)
        card_stats[num_slots] += 1

        if num_slots > max_slots:
            continue

        for c in itertools.permutations(white_cards, r=num_slots):
            sentence = CardUtils.combine(black_card, c)
            sentences.append(sentence)

    if verbose:
        time.sleep(0.01)
        print(f"Black cards (calls)       - {len(black_cards)}")
        print(f"White cards (responses)   - {len(white_cards)}")
        for i in range(1, 4):
            print(f"Cards with {i} blank(s):    - {card_stats[i]}")
        print(f"Sentences                 - {len(sentences)}")

    with open(SENTENCE_DIR, 'w+', encoding='utf-8') as f:
        f.write("\n".join(sentences))
    return sentences


def embed_sentences(model, sentences):
    if EMBED_DIR.is_file():
        return

    model = SentenceTransformer(model)
    embeddings = {}
    for s in tqdm.tqdm(sentences, desc="Embedding Sentences"):
        embeddings[s] = model.encode(s)  # <-- numpy array

    with open(EMBED_DIR, 'wb+') as f:
        pickle.dump(embeddings, f)


def main():
    download_deck()
    black_cards, white_cards = load_raw_cards()
    sentences = generate_sentences(black_cards, white_cards)
    embed_sentences('distilbert-base-nli-mean-tokens', sentences)


if __name__ == '__main__':
    main()
