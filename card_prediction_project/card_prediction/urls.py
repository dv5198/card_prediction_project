from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict, name='home'),
    path('add_draw/', views.add_draw, name='add_draw'),
]