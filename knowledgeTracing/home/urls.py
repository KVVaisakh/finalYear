from django.contrib import admin
from django.urls import path,include
from home import views

urlpatterns = [
	path('bkt/', views.bkt, name='bkt'),
	path('dktLSTM/', views.dktLSTM, name='dktLSTM'),
	path('dktGRU/', views.dktGRU, name='dktGRU'),
	path('dktMap/', views.dktMap, name='dktMap'),
]
