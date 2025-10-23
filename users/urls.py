from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('signup/', views.signup_view, name='signup'),
    path('AutoML/', views.AutoML_view, name='AutoML'),
    path('admin/user', views.admin_user_view, name='admin_user'),
]