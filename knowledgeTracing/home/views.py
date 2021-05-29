from django.shortcuts import render
from .forms import *
from .getDKT import predictDktLstm
from .getDKT import predictDktGru
import random
from shutil import copyfile
import os
from pathlib import Path
from django.contrib.auth.decorators import login_required

# Create your views here.

def bkt(request):
	return render(request, 'home/bkt.html')

@login_required
def dktLSTM(request):
	if request.method == 'POST':
		form = ModelFormWithFileField(request.POST, request.FILES)
		if form.is_valid():
			if request.FILES['file'].name.split(".")[-1] == "txt":
				location = handle_uploaded_file(request, request.FILES['file'])
				skills = predictDktLstm(location)
				return render(request, 'home/dktResult.html',{'skills':skills})
			return render(request, 'home/dkt.html', {'form': form,'model': 'LSTM'})
		else:
			return render(request, 'home/dkt.html', {'form': form,})
	else:
		form = ModelFormWithFileField()
		return render(request, 'home/dkt.html', {'form': form,'model': 'LSTM'})

@login_required
def dktGRU(request):
	if request.method == 'POST':
		form = ModelFormWithFileField(request.POST, request.FILES)
		if form.is_valid():
			if request.FILES['file'].name.split(".")[-1] == "txt":
				location = handle_uploaded_file(request, request.FILES['file'])
				skills = predictDktGru(location)
				return render(request, 'home/dktResult.html',{'skills':skills})
			return render(request, 'home/dkt.html', {'form': form,'model': 'LSTM'})
		else:
			return render(request, 'home/dkt.html', {'form': form,})
	else:
		form = ModelFormWithFileField()
		return render(request, 'home/dkt.html', {'form': form,'model': 'GRU'})

@login_required
def handle_uploaded_file(request, f):
	location = 'uploads/'+request.user.username + ".txt"
	BASE_DIR = Path(__file__).resolve().parent.parent
	if os.path.isfile(os.path.join(BASE_DIR,'uploads/')) == True:
		os.mkdir(os.path.join(BASE_DIR,'uploads/')) 
	with open(location, 'wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)
	return location

@login_required
def dktMap(request):
	skills = ["Accounting","Afrikaans","Ancient History","Anthropology","Art and Design","Applied Science","Arabic","Archaeology","Architecture","Art and Design","Agriculture","Astronomy","Astrophysics","Bengali","Biblical Hebrew","Biology","Business","Business Studies","Chemistry","Citizenship Studies","Classical Civilisation","Classical Greek","Classical Studies","Communication and Culture","Computer Science","Computing","Criminology","Critical Thinking","Civics","Dance","Design and Technology","Design and Textiles","Digital Media and Design","Digital Technology","Divinity","Drama","Drama and Theatre","Dutch","Economics","Economics and Business","Electronics","Engineering","English Language","English Language and Literature","English Literature","Environmental Science","Environmental Studies","Environmental Technology","Ethics","Fashion and Textiles","Film Studies","Food Studies","Food Technology","French","Further Mathematics","General Studies","Geography","Geology","German","Global Development","Global Perspectives and Research","Government and Politics","Greek","Gujarati","Health and Social Care","Hindi","Hinduism","History","History of Art","Home Economics","Human Biology","Humanities","ICT","Information Technology","International Relations","Irish","Islamic Studies","Italian","Latin","Law","Leisure Studies","Life and Health Sciences","language","Marine Science","Mathematics","Media Studies","Modern Hebrew","Moving Image Arts","Music","Music Technology","Mythology","Modern history","Nutrition and Food Science","Punjabi","Performance Studies","Performing Arts","Persian","Philosophy","Photography","Physical Education","Physical Science","Physics","Politics","Portuguese","Product Design","Professional Business Services","Psychology","Pure Mathematics","Quantitative Methods","Quantum physics","Quantum mechanics","Religious Studies","reegan","Science in Society","Sociology","Software Systems Development","Spanish","Sports Science","Statistics","Systems and Control Technology","Telugu","Tamil","Technology and Design","Thinking Skills","Travel and Tourism"]
	out1 = []
	out2 = []
	for i in range(125):
		if i<65:
			out1.append({"id":i, "SkillName":skills[i]})
		else:
			out2.append({"id":i, "SkillName":skills[i]})
	return render(request, 'home/dktMap.html',{'out1': out1,'out2': out2})