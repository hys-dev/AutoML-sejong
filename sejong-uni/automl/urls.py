from django.urls import path
from . import views

app_name = 'automl'

urlpatterns = [
    path('start-image-nas/', views.start_image_nas, name='start-image-nas'),
    path('end-image-nas/', views.end_image_nas, name='end-image-nas'),
    path('start-multimodal-nas/', views.start_multimodal_nas, name='start-multimodal-nas'),
    path('end-multimodal-nas/', views.end_multimodal_nas, name='end-multimodal-nas'),
    path('delete-search/', views.delete_search, name='delete-search'),
    path('start-image-retrain/', views.start_image_retrain, name='start-image-retrain'),
    path('', views.AutoML_view, name='AutoML'),
    path("api/upload-zip/", views.upload_zip, name="upload_zip"),
    path("api/upload-list/", views.get_upload_list, name="get_upload_list")
]