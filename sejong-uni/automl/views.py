import json
import os
import subprocess
import sys
import zipfile

from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from automl.models import UploadedZip, ImageNas
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

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=image_nas_dir,
                                   encoding="utf-8")
        exp_key = ""

        for line in process.stdout:
            line = line.strip()
            if line.startswith("[EXP_KEY]"):
                exp_key = line.replace("[EXP_KEY]", "").strip()

        if (exp_key != ""):
            new_image_nas = ImageNas(exp_key=exp_key, dataset_name=dataset_name, layer_candidates=layer_candidates_json,
                                 max_epochs=max_epochs,
                                 strategy=strategy, batch_size=batch_size, learning_rate=learning_rate,
                                 momentum=momentum,
                                 weight_decay=weight_decay, gradient_clip_val=gradient_clip_val, width=width,
                                 num_of_cells=num_of_cells, aux_loss_weight=aux_loss_weight)
            new_image_nas.save()

        return JsonResponse({
            "status": "ok",
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

        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads/" + category)
        os.makedirs(upload_dir, exist_ok=True)

        file_path = default_storage.save(os.path.join("uploads/" + category, file.name), file)
        #UploadedZip.objects.create(category=category, file=file_path)
        
        full_zip_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
         # 압축 해제할 폴더명 (zip 파일명 기반)
        extract_folder_name = file.name.replace(".zip", "")
        extract_folder = os.path.join(settings.MEDIA_ROOT, "uploads/" + category, extract_folder_name)
        
        # ZIP 압축 해제
        with zipfile.ZipFile(full_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # 이미지 파일 목록 수집 (jpg/png/jpeg만)
        allowed_ext = [".jpg", ".jpeg", ".png"]
        preview_files = [
            f"/media/uploads/{extract_folder_name}/{f}"
            for f in os.listdir(extract_folder)
            if os.path.splitext(f)[1].lower() in allowed_ext
        ]

        # DB 저장
        #UploadedZip.objects.create(category=category, file=file_path)

        return JsonResponse({
            "success": True,
            "filename": file.name,
            "images": preview_files,  # 프론트에서 바로 미리보기 가능
        })

    return JsonResponse({"success": False, "error": "No file uploaded"}, status=400)


def get_upload_list(request):
    category = request.GET.get("category", "image")
    files = UploadedZip.objects.filter(category=category).order_by("-uploaded_at")
    data = [{"filename": os.path.basename(f.file.name), "uploaded_at": f.uploaded_at.strftime("%Y-%m-%d %H:%M:%S")} for f in files]
    return JsonResponse({"files": data})