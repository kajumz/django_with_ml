from django.urls import path
from django.views.generic import TemplateView
from mlapp import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('', views.index, name='index'),
    path('input', views.home_view),

]

