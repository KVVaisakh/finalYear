from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.http import HttpResponseRedirect,HttpResponse
from .forms import *
from .getSentiment import predict
from django.shortcuts import redirect
import shutil
import time
import os

def index(request):
	if request.method == 'POST':
		if request.user.is_authenticated == False:
			return redirect('login')
		form = SentimentForm(request.POST)
		if form.is_valid():
			sentiment = form.cleaned_data
			message = predict(sentiment['message'])
			if message == "Negative":
				src="uploads/"+request.user.username+".txt"
				dst="saved/"+request.user.username+str(time.time())+".txt"
				if os.path.isfile(src) == True:
					shutil.copy(src, dst)
			return render(request, 'knowledgeTracing/sentimentResult.html',{'message':message})
		else:
			return redirect('index')
	form = SentimentForm()
	loggedIn = "No"
	if request.user.is_authenticated:
		loggedIn = "Yes"
	print(loggedIn)
	print(request.user.is_authenticated)
	return render(request, 'knowledgeTracing/index.html',{'form' : form, 'loggedIn':loggedIn})

def logIn(request):
	if request.user.is_authenticated:
		return redirect('index')
	if request.method == 'POST':
		form=LoginForm(request.POST)
		if form.is_valid():
			user = form.cleaned_data
			username = user['username']
			password =  user['password']
			user = authenticate(username = username, password = password)
			if user:
				login(request,user)
				print("Success")
				return redirect('index')
	form=LoginForm()
	return render(request, 'knowledgeTracing/login.html', {'form':form})


def signup(request):
	if request.method == 'POST':
		form = UserRegistrationForm(request.POST)
		if form.is_valid():
			user = form.cleaned_data
			username = user['username']
			password =  user['password']
			email = user['email']
			if not User.objects.filter(username=username).exists():
				User.objects.create_user(username, email, password)
				user = authenticate(username = username, password = password)
				return redirect('index')
			else:
				return render(request, 'knowledgeTracing/signup.html', {'form' : form, 'error':"username already exists"})
	else:
		form = UserRegistrationForm()
	return render(request, 'knowledgeTracing/signup.html', {'form' : form})