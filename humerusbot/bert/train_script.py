import numpy as np
import pickle5

from sentence_transformers import SentenceTransformer

from utils import dataManager
from utils import pcaifier
from utils import humerus

postgres_keys = "/content/postgres_access.txt"
dm = dataManager.DataManager(postgres_keys)
game_results = dm.cleanGameData(dm.fetchGameData())

pca_cache_path = "../data/pca_cache.pkl"
embed_table_path = "../data/embed_table.pkl"




s_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

with open(embed_table_path, 'rb') as f:
    embed_table = pickle5.load(f)

embed_dict = {}
for i, x in enumerate(zip(embed_table['sentence'], embed_table['Bert_embed_69'])):
    embed_dict[x[0]] = x[1]

pca_convert = pcaifier.pca(pca_cache_path)

raw_play_strings = [i[0] for i in game_results]
# 1 if win, 0 if loss
y_data = [1 if i[1] else 0 for i in game_results]
X_train, Y_train = [], []
dictkeys = list(embed_dict.keys())
for i, x in enumerate(raw_play_strings):
    try:
        embed = embed_dict[x]
        X_train.append(embed)
        Y_train.append(y_data[i])
    except:
        embed = pca_convert.reduce_kdim(s_model.encode(x), 69)
        X_train.append(embed)
        Y_train.append(y_data[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)


# change to from_file once we get a first one trained up
pred_net = humerus.predictionNetwork.from_scratch(69)

pred_net.train_model(X_train, Y_train, 69)
pred_net.save_models("../models/m1")














