from django.urls import path
from . import views

app_name = 'automl'

urlpatterns = [
    path('start-image-nas/', views.start_image_nas, name='start-image-nas'),
]