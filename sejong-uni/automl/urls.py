from django.urls import path
from . import views

app_name = 'automl'

urlpatterns = [
    path('start-image-nas/', views.start_image_nas, name='start-image-nas'),
    path('end-image-nas/', views.end_image_nas, name='end-image-nas'),
    path('', views.AutoML_view, name='AutoML'),
    path("api/upload-zip/", views.upload_zip, name="upload_zip"),
    path("api/upload-list/", views.get_upload_list, name="get_upload_list")
]