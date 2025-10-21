from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from .models import *
from .forms import *

# Create your views here.
from django.http import HttpResponse

def index(request):
    return HttpResponse("hello world")