import pickle
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
# Load the model from the file
with open('D:\django_with_ml\logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
print('input')
text = str(input())
print('tokenize')
tokens = tokenizer.encode(text, return_tensors='pt', truncation=True)
print('get_embed')
with torch.no_grad():
    last_hidden_states = bert_model(tokens)
embed = last_hidden_states.last_hidden_state
features = embed[0].mean(dim=0).tolist()
test = np.array(features)
test = test.reshape(1, -1)
prediction = model.predict(test)
print(prediction)

