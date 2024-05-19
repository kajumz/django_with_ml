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
    #f request.method == 'POST':
    #    form = TextInputForm(request.POST)
    #    if form.is_valid():
    #        text = form.cleaned_data['text']
    #        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True)
    #        with torch.no_grad():
    #            last_hidden_states = bert_model(tokens)
    #        embed = last_hidden_states.last_hidden_state
    #        features = embed[0].mean(dim=0).tolist()
    #        test = np.array(features)
    #        test = test.reshape(1, -1)
    #        prediction = model.predict(test)
    #        return JsonResponse({'prediction': prediction[0]})
    #else:
    #    form = TextInputForm()
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
            prediction = model.predict(test)
            return HttpResponse(prediction[0])
    else:
        form = TextForm()
    data = {
        'form': form

    }
    return render(request, 'predict.html', data)


def home_view(request):
    context = {}
    context['form'] = InputForm()
    return render(request, "home.html", context)
