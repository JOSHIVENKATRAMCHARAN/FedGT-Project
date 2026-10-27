from django.urls import path

from . import views

urlpatterns = [
    path('upload/', views.upload_dataset, name='upload_dataset'),
]
