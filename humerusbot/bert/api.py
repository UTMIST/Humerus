from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import json
import pickle5

from utils import humerus
from utils import pcaifier

from sentence_transformers import SentenceTransformer


pca_cache_path = "../../data/pca_cache_69_5050.pkl"
embed_table_path = "../../data/training_set_data_69_5050.pkl"


with open(embed_table_path, 'rb') as f:
    embed_table = pickle5.load(f)

embed_dict = {}
for i, x in enumerate(zip(embed_table['sentence'], embed_table['Bert_embed_69'])):
    embed_dict[x[0]] = x[1]

pca_convert = pcaifier.pca(pca_cache_path)

s_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# pred_net = humerus.predictionNetwork.from_scratch(69)
pred_net = humerus.predictionNetwork.from_file(69, "./models/m1.h5")



app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
@app.route('/', methods=['GET', 'POST']) 
def eval_model():
    global embed_dict, s_model, pca_convert, pred_net

    if request.method == 'POST':  
        #POST a json list only. for 1 element, do 1 elem list.
        raw_text = request.form.get('text')
        if type(raw_text) == str:
            raw_text = json.loads(raw_text.replace("'", '"'))
        try:
            embed = embed_dict[raw_text]
        except:
            embed = pca_convert.reduce_kdim(s_model.encode(raw_text), 69)
        
        payload = pred_net.evaluate(embed)
        return jsonify([str(i) for i in payload.flatten()])
        
    s = """<form method="POST">
                  Text: <input type="text" name="text"><br>
                  <input type="submit" value="Submit"><br>
              </form>"""
    return s








