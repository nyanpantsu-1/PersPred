from django.shortcuts import render
from django.http import HttpResponse
from polls import predict_main as pm
from .forms import SearchForm

def index(request):
    
    return HttpResponse("in polls")

def first(request):
    return HttpResponse("polls for First")

def search(request):
    form = SearchForm()
    if request.method == 'GET':
        form = SearchForm(request.GET)
        if form.is_valid():
            query = form.cleaned_data['query']
            results=pm.predict_logistic()
            return render(request, 'polls/results.html', {'query': query, 'results': results})
    return render(request, 'polls/index.html', {'form': form})