# mlapp/views.py
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import TextForm
from .forms import InputForm
from django.http import JsonResponse
#from mlapp.forms import TextInputForm
from mlapp.ml_models import model, tokenizer, bert_model
import torch
import numpy as np



def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def predict(request):
    prediction = None
    if request.method == 'POST':
        form = TextForm(request.POST)
        if form.is_valid():

            text = form.cleaned_data['message']
            tokens = tokenizer.encode(text, return_tensors='pt', truncation=True)
            with torch.no_grad():
                last_hidden_states = bert_model(tokens)
            embed = last_hidden_states.last_hidden_state
            features = embed[0].mean(dim=0).tolist()
            test = np.array(features)
            test = test.reshape(1, -1)
            pred = model.predict(test)
            if pred[0] == 0:
                prediction = 'a'
            elif pred[0] == 1:
                prediction = 'b'
            elif pred[0] == 2:
                prediction = 'c'
            elif pred[0] == 3:
                prediction = 'd'
            #return HttpResponse(prediction[0])
    else:
        form = TextForm()
    data = {
        'form': form,
        'prediction': prediction

    }
    return render(request, 'predict.html', data)


def home_view(request):
    context = {}
    context['form'] = InputForm()
    return render(request, "home.html", context)
