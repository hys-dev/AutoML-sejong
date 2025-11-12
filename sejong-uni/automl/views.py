import json
import os
import subprocess
import sys
import zipfile
import redis
import threading

from django.core.files.storage import default_storage
from django.db import close_old_connections
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from automl.models import UploadedZip, ImageNas, MultimodalNas
from config import settings

# Create your views here.

processes = {}

def index(request):
    return HttpResponse("hello world")

@csrf_exempt
def start_image_nas(request):
    print("views.py start_image_nas")
    user_id = "0"
    if request.user.is_authenticated:
        user_id = request.user.username

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

    image_nas_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../image_nas"))
    nas_py = os.path.join(image_nas_dir, "nas.py")

    print("nas.py: ", os.path.exists(nas_py))

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

    exp_key_container = {"exp_key": None}  # [EXP_KEY]를 저장할 dict

    def run_nas_background(pid, exp_key_container):
        """
        NAS 실행 후 종료 시 DB 업데이트
        """
        process = processes[pid]

        process.wait()  # NAS 끝날 때까지 대기
        close_old_connections()

        if exp_key_container["exp_key"]:
            ImageNas.objects.filter(exp_key=exp_key_container["exp_key"]).update(
                end_time=timezone.now()
            )

        processes.pop(pid, None)
        print(f"NAS process {pid} finished.")


    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=image_nas_dir,
                                   bufsize=1, encoding="utf-8", universal_newlines=True)
        pid = process.pid
        processes[pid] = process
        exp_key = None

        for line in process.stdout:
            print(line, end="")  # 터미널에 실시간 출력
            if line.startswith("[EXP_KEY]"):
                exp_key = line.replace("[EXP_KEY]", "").strip() or None
                print("Detected exp_key:", exp_key)
                exp_key_container["exp_key"] = exp_key
                break

        if exp_key:
            new_image_nas = ImageNas(exp_key=exp_key, dataset_name=dataset_name, layer_candidates=layer_candidates_json,
                                 max_epochs=max_epochs, user_id=user_id,
                                 strategy=strategy, batch_size=batch_size, learning_rate=learning_rate,
                                 momentum=momentum,
                                 weight_decay=weight_decay, gradient_clip_val=gradient_clip_val, width=width,
                                 num_of_cells=num_of_cells, aux_loss_weight=aux_loss_weight)
            new_image_nas.save()

            threading.Thread(target=run_nas_background, args=(pid, exp_key_container), daemon=True).start()

            return JsonResponse({
                "status": "started",
                "pid": pid,
                "exp_key": exp_key
            })

    except Exception as e:
        print("error", e)
        return JsonResponse({
            "status": "error",
            "message": str(e),
        })

@csrf_exempt
def start_multimodal_nas(request):
    print("views.py start_multimodal_nas")

    user_id = "0"
    if request.user.is_authenticated:
        user_id = request.user.username

    dataset_name = request.POST.get("dataset_name")
    max_epochs = request.POST.get('max_epochs')
    learning_rate = request.POST.get('learning_rate')
    min_learning_rate = request.POST.get('min_learning_rate')
    warmup_epochs = request.POST.get('warmup_epochs')
    batch_size = request.POST.get('batch_size')
    weight_decay = request.POST.get('weight_decay')
    optimizer = request.POST.get('optimizer')
    lr_scheduler = request.POST.get('lr_scheduler')

    multimodal_nas_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../multimodal_nas"))
    multimodal_nas_py = os.path.join(multimodal_nas_dir, "nas.py")

    cmd = [
        sys.executable, multimodal_nas_py,
        str(dataset_name),
        str(max_epochs),
        str(learning_rate),
        str(min_learning_rate),
        str(warmup_epochs),
        str(batch_size),
        str(weight_decay),
        str(optimizer),
        str(lr_scheduler),
    ]

    exp_key_container = {"exp_key": None}  # [EXP_KEY]를 저장할 dict

    def run_nas_background(pid, exp_key_container):
        """
        NAS 실행 후 종료 시 DB 업데이트
        """
        process = processes[pid]

        process.wait()  # NAS 끝날 때까지 대기
        close_old_connections()

        if exp_key_container["exp_key"]:
            MultimodalNas.objects.filter(exp_key=exp_key_container["exp_key"]).update(
                end_time=timezone.now()
            )

        processes.pop(pid, None)
        print(f"NAS process {pid} finished.")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=multimodal_nas_dir,
                                   bufsize=1, encoding="utf-8", universal_newlines=True)
        pid = process.pid
        processes[pid] = process
        exp_key = None

        for line in process.stdout:
            print(line, end="")  # 터미널에 실시간 출력
            if line.startswith("[EXP_KEY]"):
                exp_key = line.replace("[EXP_KEY]", "").strip() or None
                print("Detected exp_key:", exp_key)
                exp_key_container["exp_key"] = exp_key
                break

        if exp_key:
            new_multimodal_nas = MultimodalNas(exp_key=exp_key, dataset_name=dataset_name,max_epochs=max_epochs, user_id=user_id,
                                          batch_size=batch_size, learning_rate=learning_rate, min_learning_rate=min_learning_rate,
                                          weight_decay=weight_decay, optimizer=optimizer, lr_scheduler=lr_scheduler)
            new_multimodal_nas.save()

            threading.Thread(target=run_nas_background, args=(pid, exp_key_container), daemon=True).start()

            return JsonResponse({
                "status": "started",
                "pid": pid,
                "exp_key": exp_key
            })

    except Exception as e:
        print("error", e)
        return JsonResponse({
            "status": "error",
            "message": str(e),
        })

@csrf_exempt
def end_image_nas(request):
    print("views.py end_image_nas")
    pid = int(request.POST.get("pid"))
    exp_key = request.POST.get("exp_key")
    process = processes.get(pid)
    if process and process.poll() is None:
        process.kill()
        ImageNas.objects.filter(exp_key=exp_key).update(end_time=timezone.now())
        processes.pop(pid, None)
        return JsonResponse({"status": "terminated"})
    else:
        return JsonResponse({"status": "not_running"})

def AutoML_view(request):
    #추후 로그인한 사용자만 접근 가능하도록 수정
    user_id = "0"
    if request.user.is_authenticated:
        user_id = request.user.username

    obj_image = ImageNas.objects.filter(user_id=user_id)
    obj_multi = MultimodalNas.objects.filter(user_id=user_id)
    return render(request, 'detail/AutoML.html', {'imageNas': obj_image, 'multimodalNas': obj_multi})

@csrf_exempt
def upload_zip(request):
    if request.method == "POST" and request.FILES.get("file"):
        category = request.POST.get("category", "image")
        file = request.FILES["file"]
        
        relative_path = f"uploads/{category}/{file.name}"
        zip_path = default_storage.save(relative_path, file)
        full_zip_path = os.path.join(settings.MEDIA_ROOT, zip_path)

        #UploadedZip.objects.create(category=category, file=file_path)
        
         # 압축 해제할 폴더명 (zip 파일명 기반)
        extract_folder_name = file.name.replace(".zip", "")
        extract_folder = os.path.join(settings.MEDIA_ROOT, "uploads", category, extract_folder_name)
        os.makedirs(extract_folder, exist_ok=True)
        
        # ZIP 압축 해제
        with zipfile.ZipFile(full_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # 이미지 파일 목록 수집 (jpg/png/jpeg만)
        allowed_ext = [".jpg", ".jpeg", ".png"]
        preview_files = []
        for root, _, files in os.walk(extract_folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in allowed_ext:
                    rel_path = os.path.relpath(os.path.join(root, f), settings.MEDIA_ROOT)
                    rel_path = rel_path.replace('\\', '/')
                    preview_files.append(f"/media/{rel_path}")

        preview_files = preview_files[:30]
        
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