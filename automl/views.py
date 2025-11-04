import json
import os
import subprocess
import sys

from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from automl.models import UploadedZip
from config import settings

# Create your views here.

def index(request):
    return HttpResponse("hello world")

@csrf_exempt
def start_image_nas(request):
    print("views.py start_image_nas")
    dataset_name = request.POST.get("dataset_name")
    layer_candidates = request.POST.getlist('layer_candidates[]')  # 배열일 경우 getlist로 받고 이름 뒤에 꼭 [] 표시
    max_epochs = request.POST.get('max_epochs')
    strategy = request.POST.get('strategy')
    batch_size = request.POST.get('batch_size')
    learning_rate = request.POST.get('learning_rate')
    momentum = request.POST.get('momentum')
    weight_decay = request.POST.get('weight_decay')
    gradient_clip_val = request.POST.get('gradient_clip')
    width = request.POST.get('width')
    num_of_cells = request.POST.get('num_of_cells')
    aux_loss_weight = request.POST.get('aux_loss_weight')

    print(dataset_name)

    layer_candidates_json = json.dumps(layer_candidates)


    try:
        print("subprocess.run")
        image_nas_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../image_nas"))
        nas_py = os.path.join(image_nas_dir, "nas.py")

        cmd = [
            sys.executable, nas_py,
            str(dataset_name),
            layer_candidates_json,
            str(max_epochs),
            str(strategy),
            str(batch_size),
            str(learning_rate),
            str(momentum),
            str(weight_decay),
            str(gradient_clip_val),
            str(width),
            str(num_of_cells),
            str(aux_loss_weight),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=image_nas_dir, encoding="utf-8")

        print("stdout: ", result.stdout)
        print("stderr: ", result.stderr)

        return JsonResponse({
            "status": "ok",
            "stdout": result.stdout,
            "stderr": result.stderr,
        })

    except Exception as e:
        print("error", e)
        return JsonResponse({
            "status": "error",
            "message": str(e),
        })


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