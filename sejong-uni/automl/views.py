import json
import os
import platform
import subprocess
import sys
import zipfile
import threading
import signal
import random
import string

from django.core.files.storage import default_storage
from django.core.paginator import Paginator
from django.core.cache import cache
from django.db import close_old_connections
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from automl.models import UploadedZip, ImageNas, MultimodalNas, MultimodalEvo
from config import settings

# Create your views here.

def kill_process(pid):
    try:
        if platform.system() == "Windows":
            os.kill(pid, signal.CTRL_C_EVENT)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


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

    def run_nas(pid, exp_key_container):
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

        close_old_connections()

        exp_key = exp_key_container["exp_key"]
        if exp_key:
            ImageNas.objects.filter(exp_key=exp_key).update(
                end_time=timezone.now()
            )

        cache.delete(f"nas:image:{pid}")
        print(f"NAS process {pid} finished.")


    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=image_nas_dir,
                                   bufsize=1, encoding="utf-8", universal_newlines=True)
        pid = process.pid
        exp_key = None

        for line in process.stdout:
            print(line, end="")  # 터미널에 실시간 출력
            if line.startswith("[EXP_KEY]"):
                exp_key = line.replace("[EXP_KEY]", "").strip()
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

            cache.set(f"nas:image:{pid}",
                      json.dumps({
                          "pid": pid,
                          "exp_key": exp_key,
                          "status": "running",
                      }),
                      timeout=None
                      )

            threading.Thread(target=run_nas, args=(pid, exp_key_container), daemon=True).start()

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

    evo_max_epochs = request.POST.get('evo_max_epochs')
    evo_batch_size = request.POST.get('evo_batch_size')
    evo_min_param_limits = request.POST.get('evo_min_param_limits')
    evo_param_limits = request.POST.get('evo_param_limits')
    evo_population_num = request.POST.get('evo_population_num')
    evo_select_num = request.POST.get('evo_select_num')
    evo_crossover_num = request.POST.get('evo_crossover_num')
    evo_mutation_num  = request.POST.get('evo_mutation_num')

    multimodal_nas_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../multimodal_nas"))
    nas_py = os.path.join(multimodal_nas_dir, "nas.py")
    evo_py = os.path.join(multimodal_nas_dir, "evo.py")

    nas_cmd = [
        sys.executable, nas_py,
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

    def run_evo(nas_exp_key):
        evo_cmd = [
            sys.executable, evo_py,
            str(nas_exp_key),
            str(dataset_name),
            str(evo_max_epochs),
            str(evo_batch_size),
            str(evo_min_param_limits),
            str(evo_param_limits),
            str(evo_select_num),
            str(evo_population_num),
            str(evo_crossover_num),
            str(evo_mutation_num),
        ]

        evo_process = subprocess.Popen(evo_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, cwd=multimodal_nas_dir)
        evo_pid = evo_process.pid
        evo_exp_key = None

        for line in evo_process.stdout:
            print(line, end="")  # 터미널에 실시간 출력
            if line.startswith("[EXP_KEY]"):
                evo_exp_key = line.replace("[EXP_KEY]", "").strip() or None
                print("Detected exp_key:", evo_exp_key)
                exp_key_container["evo_exp_key"] = evo_exp_key
                break

        cache_key = f"nas:multi:{nas_exp_key}"
        obj = cache.get(cache_key)
        if obj:
            cache.set(cache_key, {
                "nas_pid": nas_pid,
                "nas_exp_key": nas_exp_key,
                "evo_pid": evo_pid,
                "evo_exp_key": evo_exp_key,
                "current_pid": evo_pid,
                "status": "evo_running"
            }, timeout=None)

        evo_process.wait()
        close_old_connections()

        MultimodalEvo.objects.create(
            exp_key=evo_exp_key,
            multimodal_nas_id=MultimodalNas.objects.filter(exp_key=nas_exp_key)[0].pk,
            nas_exp_key=nas_exp_key,
            max_epochs=evo_max_epochs,
            batch_size=evo_batch_size,
            min_param_limits=evo_min_param_limits,
            param_limits=evo_param_limits,
            population_num=evo_population_num,
            select_num=evo_select_num,
            crossover_num=evo_crossover_num,
            mutation_num=evo_mutation_num,
            population=evo_population_num,
        )

        cache.delete(cache_key)
        print(f"[EVO done] pid={evo_pid}")

    def run_nas(nas_pid, exp_key_container):
        try:
            os.waitpid(nas_pid, 0)
        except Exception:
            pass

        close_old_connections()

        exp_key = exp_key_container["nas_exp_key"]
        if not exp_key:
            return

        MultimodalNas.objects.filter(exp_key=exp_key).update(
            end_time = timezone.now(),
        )

        print(f"[NAS done] pid={nas_pid}. Starting EVO...")

        # --- EVO 시작 ---
        threading.Thread(
            target=run_evo,
            args=(exp_key,),
            daemon=True
        ).start()
    # end run_nas

    try:
        nas_process = subprocess.Popen(nas_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=multimodal_nas_dir,
                                   encoding="utf-8")
        nas_pid = nas_process.pid
        nas_exp_key = None

        for line in nas_process.stdout:
            print(line, end="")  # 터미널에 실시간 출력
            if line.startswith("[EXP_KEY]"):
                nas_exp_key = line.replace("[EXP_KEY]", "").strip() or None
                print("Detected exp_key:", nas_exp_key)
                exp_key_container["nas_exp_key"] = nas_exp_key
                break

        if nas_exp_key:
            MultimodalNas.objects.create(exp_key=nas_exp_key, dataset_name=dataset_name,max_epochs=max_epochs, user_id=user_id,
                                          batch_size=batch_size, learning_rate=learning_rate, min_learning_rate=min_learning_rate,
                                          weight_decay=weight_decay, optimizer=optimizer, lr_scheduler=lr_scheduler)

            cache.set(
                f"nas:multi:{nas_exp_key}",
                {
                    "nas_pid": nas_pid,
                    "nas_exp_key": nas_exp_key,
                    "evo_pid": None,
                    "evo_exp_key": None,
                    "current_pid": nas_pid,
                    "current_exp_key": nas_exp_key,
                    "status": "nas_running",
                },
                timeout=None
            )

            threading.Thread(target=run_nas, args=(nas_pid, exp_key_container), daemon=True).start()

            return JsonResponse({
                "status": "nas_started",
                "nas_pid": nas_pid,
                "exp_key": nas_exp_key
            })
            
        else:
            return JsonResponse({
            "status": "error",
            "message": "EXP_KEY not received from nas.py"
        }, status=500)

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

    cache_key = f"nas:image:{pid}"
    data = cache.get(cache_key)

    if not data:
        return JsonResponse({
            "status": "not running",
        })

    try:
        kill_process(pid)
    except ProcessLookupError:
        pass
    except PermissionError:
        print("No permission to kill this process")
    except Exception as e:
        print("kill error", e)

    ImageNas.objects.filter(exp_key=exp_key).update(end_time=timezone.now())

    cache.delete(cache_key)

    return JsonResponse({"status": "terminated"})

def end_multimodal_nas(request):
    nas_exp_key = request.POST.get("nas_exp_key")
    cache_key = f"nas:multi:{nas_exp_key}"

    data = cache.get(cache_key)
    if not data:
        return JsonResponse({"status": "not_running"})

    current_pid = data.get("current_pid")

    if current_pid:
        kill_process(current_pid)

        if current_pid == data.get("nas_pid"):
            MultimodalNas.objects.filter(exp_key=nas_exp_key).update(
                end_time = timezone.now(),
            )
        elif current_pid == data.get("evo_pid") :
            MultimodalEvo.objects.filter(nas_exp_key=nas_exp_key).update(
                end_time=timezone.now()
            )

    # 상태를 finished로 업데이트
    data["status"] = "stopped"
    data["current_pid"] = None
    data["current_exp_key"] = None
    cache.set(cache_key, data, timeout=None)

    return JsonResponse({"status": "stopped"})

def start_image_retrain(request):
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
    retrain_py = os.path.join(image_nas_dir, "retrain.py")

    cmd = [
        sys.executable, retrain_py,
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

def delete_search(request):
    print("views.py delete_search")

def AutoML_view(request):
    #추후 로그인한 사용자만 접근 가능하도록 수정
    user_id = "0"
    if request.user.is_authenticated:
        user_id = request.user.username

    obj_image = ImageNas.objects.filter(user_id=user_id).order_by('-image_nas_id')

    paginator_image = Paginator(obj_image, 5)
    page_number_image = request.GET.get("image_page")
    page_obj_image = paginator_image.get_page(page_number_image)

    obj_multi = MultimodalNas.objects.filter(user_id=user_id).order_by('-multimodal_nas_id')

    paginator_multi = Paginator(obj_multi, 5)
    page_number_multi = request.GET.get("multi_page")
    page_obj_multi = paginator_multi.get_page(page_number_multi)

    return render(request, 'detail/AutoML.html', {'imageNas': obj_image, 'multimodalNas': obj_multi,
                                                  'page_obj_image': page_obj_image, 'page_obj_multi': page_obj_multi })

def Retrain_view(request):
    return render(request, 'detail/retrain.html')

def Hyperparameter_view(request):
    return render(request, 'detail/hyperparameter.html')

@csrf_exempt
def upload_zip(request):
    if request.method == "POST" and request.FILES.get("file"):
        category = request.POST.get("category")
        file = request.FILES["file"]
        chars = string.ascii_letters + string.digits   # 영문 대소문자 + 숫자
        result = ''.join(random.choices(chars, k=8))
        file_name = file.name.replace(".zip", "") + '_' + result;
        
        relative_path = f"uploads/{category}/{file_name} + 'zip'"
        zip_path = default_storage.save(relative_path, file)
        full_zip_path = os.path.join(settings.MEDIA_ROOT, zip_path)
        
         # 압축 해제할 폴더명 (zip 파일명 기반)
        extract_folder_name = file.name.replace(".zip", "")
        extract_folder = os.path.join(settings.MEDIA_ROOT, "uploads", category, file_name + '.zip')
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
        #UploadedZip.objects.create(category=category, file=file_path, file_name)

        return JsonResponse({
            "success": True,
            "filename": file_name,
            "images": preview_files,  # 프론트에서 바로 미리보기 가능
        })

    return JsonResponse({"success": False, "error": "No file uploaded"}, status=400)


def get_upload_list(request):
    category = request.GET.get("category")
    files = UploadedZip.objects.filter(category=category).order_by("-uploaded_at")
    data = [{"filename": os.path.basename(f.file.name), "uploaded_at": f.uploaded_at.strftime("%Y-%m-%d %H:%M:%S")} for f in files]
    return JsonResponse({"files": data})

def upload_zip_delete(request):
    category = request.POST.get("category")
    file_name = request.POST.get("file_name")
    file_path = f"uploads/midea/{category}/{file_name}.zip"
    
    if os.path.exists(file_path):
        os.remove(file_path)
        print("삭제 완료")
    else:
        print("파일이 존재하지 않습니다.")