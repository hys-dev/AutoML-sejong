from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from users.forms import UserForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
from .models import UploadedZip
import os, json


# Create your views here.
def login_view(request):
    return render(request, 'users/login.html')

def logout_view(request):
    logout(request)
    return redirect('index')

def signup_view(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)  # 사용자 인증
            login(request, user)  # 로그인
            return redirect('index')
    else:
        form = UserForm()
    return render(request, 'users/signup.html', {'form': form})

def AutoML_view(request):
    return render(request, 'detail/AutoML.html')

@csrf_exempt
def upload_zip(request):
    if request.method == "POST" and request.FILES.get("file"):
        category = request.POST.get("category", "image")
        file = request.FILES["file"]

        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        file_path = default_storage.save(os.path.join("uploads", file.name), file)
        UploadedZip.objects.create(category=category, file=file_path)
        return JsonResponse({"success": True, "filename": file.name, "category": category})
    return JsonResponse({"success": False, "error": "No file uploaded"}, status=400)


def get_upload_list(request):
    category = request.GET.get("category", "image")
    files = UploadedZip.objects.filter(category=category).order_by("-uploaded_at")
    data = [{"filename": os.path.basename(f.file.name), "uploaded_at": f.uploaded_at.strftime("%Y-%m-%d %H:%M:%S")} for f in files]
    return JsonResponse({"files": data})