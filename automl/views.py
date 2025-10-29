from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


# Create your views here.

def index(request):
    return HttpResponse("hello world")

@csrf_exempt
def start_image_nas(request):
    if request.method == "POST":
        layer_candidates = request.POST.getlist('layer_candidates[]')  # 배열일 경우 getlist로 받고 이름 뒤에 꼭 [] 표시
        max_epochs = request.POST.get('max_epochs')
        batch_size = request.POST.get('batch_size')
        learning_rate = request.POST.get('learning_rate')
        momentum = request.POST.get('momentum')
        weight_decay = request.POST.get('weight_decay')
        gradient_clip = request.POST.get('gradient_clip')
        width = request.POST.get('width')
        num_of_cells = request.POST.get('num_of_cells')
        aux_loss_weight = request.POST.get('aux_loss_weight')

    return JsonResponse({"success": True})

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