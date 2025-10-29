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