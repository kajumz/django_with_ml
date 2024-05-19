import pickle
from transformers import BertTokenizer, BertModel
# Load the model from the file
with open('D:\django_with_ml\logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')



