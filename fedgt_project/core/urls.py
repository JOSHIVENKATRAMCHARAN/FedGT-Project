from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('', views.dashboard, name='dashboard'),
    path('clients/', views.client_monitor, name='client_monitor'),
    path('groups/', views.group_viewer, name='group_viewer'),
    path('training/', views.training_dashboard, name='training_dashboard'),
    path('models/', views.ml_models, name='ml_models'),
    path('attacks/', views.attack_analyzer, name='attack_analyzer'),
    path('detection/', views.detection, name='detection'),
    path('reports/', views.reports, name='reports'),
    path('fedgt/', views.fedgt_dashboard, name='fedgt_dashboard'),
]
