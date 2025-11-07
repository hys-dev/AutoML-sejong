let charts = {};

async function uploadFile(inputId, category) {
  const input = document.getElementById(inputId);
  const file = input.files[0];
  if (!file) return alert('파일을 선택해주세요.');

  const formData = new FormData();
  formData.append('file', file);
  formData.append('category', category);

  const res = await fetch('/automl/api/upload-zip/', { method: 'POST', body: formData });
  const data = await res.json();

  if (data.success) {
    alert(`[${category}] 업로드 성공: ${data.filename}`);
    //loadUploadList(category);
  } else {
    alert('업로드 실패');
  }

  // 이미지 미리보기 영역
  const containerId = category === 'image' ? 'image-preview' : 'multi-preview';
  renderImageGrid(containerId, data.images);
}

async function loadUploadList(category) {
  const res = await fetch('/automl/api/upload-list/?category=${category}');
  const data = await res.json();
  const listId = category === 'image' ? 'image-upload-list' : 'multi-upload-list';
  const container = document.getElementById(listId);
  container.innerHTML = data.files.map(f => `
    <li class="flex justify-between border-b py-1">
      <span>${f.filename}</span>
      <span class="text-xs text-gray-500">${f.uploaded_at}</span>
    </li>`).join('');
}

function renderImageGrid(containerId, imageList) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';

  // 최대 30개까지만 표시
  const maxCount = 30;
  const images = imageList.slice(0, maxCount);

  images.forEach(src => {
    const img = document.createElement('img');
    img.src = src;
    img.className =
      'w-20 h-20 object-cover rounded border hover:scale-110 transition-transform duration-200';
    container.appendChild(img);
  });

  // 남은 이미지가 있으면 “+더보기” 표시
  if (imageList.length > maxCount) {
    const moreDiv = document.createElement('div');
    moreDiv.className =
      'w-20 h-20 flex items-center justify-center bg-gray-200 rounded border text-gray-600 cursor-pointer hover:bg-gray-300';
    moreDiv.textContent = `+${imageList.length - maxCount}`;
    moreDiv.onclick = () => {
      renderImageGrid(containerId, imageList); // 클릭 시 전체 다시 렌더링
    };
    container.appendChild(moreDiv);
  }
}

// 초기 로드
//loadUploadList('image');
//loadUploadList('multi');

function showTab(tab) {
    document.getElementById('panel-image').classList.add('hidden');
    document.getElementById('panel-multi').classList.add('hidden');

    document.getElementById('tab-image').classList.remove('tab-active');
    document.getElementById('tab-image').classList.add('tab-inactive');

    document.getElementById('tab-multi').classList.remove('tab-active');
    document.getElementById('tab-multi').classList.add('tab-inactive');

    if(tab==='image') {
    document.getElementById('panel-image').classList.remove('hidden');
    document.getElementById('tab-image').classList.add('tab-active');
    document.getElementById('tab-image').classList.remove('tab-inactive');
    initChart('image-chart');
    } else {
    document.getElementById('panel-multi').classList.remove('hidden');
    document.getElementById('tab-multi').classList.add('tab-active');
    document.getElementById('tab-multi').classList.remove('tab-inactive');
    initChart('multi-chart');
    }
}


function initChart(id){
    const el = document.getElementById(id);
    if(!el || charts[id]) return;
        const ctx = el.getContext('2d');
        charts[id] = new Chart(ctx, {
        type: 'line',
        data: { labels: ['1','2','3','4','5'], datasets: [{label: 'Accuracy', data:[20,40,55,70,80], tension:0.3, borderColor:'#2563eb'}] },
        options: { responsive:true, maintainAspectRatio:false }
    });
}


function startSearch(kind){
    /*
    const logEl = document.getElementById(kind==='image'?'image-log':'multi-log');
    logEl.textContent='';
    let i=0;
    const interval = setInterval(()=>{
        i++;
        logEl.textContent += `Epoch ${i}: acc=${Math.round(30+Math.random()*60)}%\n`;
        logEl.scrollTop=logEl.scrollHeight;
        if(i>=5) clearInterval(interval);
    },800);
    */
    const layer_candidates = document.querySelectorAll('input[name="layer_candidates"]');
    const selected = [];

    layer_candidates.forEach(candidate => {
        if(candidate.checked) {
            selected.push(candidate.value);
        }
    });

    console.log(selected);

    let dataset_name = "cifar10";
    let strategy = document.getElementById('strategy').value;
    let max_epochs = document.querySelector('input[name="max_epochs"]').value;
    let batch_size = document.querySelector('input[name="batch_size"]').value;
    let learning_rate = document.querySelector('input[name="learning_rate"]').value;
    let momentum = document.querySelector('input[name="momentum"]').value;
    let weight_decay = document.querySelector('input[name="weight_decay"]').value;
    let gradient_clip = document.querySelector('input[name="gradient_clip"]').value;
    let width = document.querySelector('input[name="width"]').value;
    let num_of_cells = document.querySelector('input[name="num_of_cells"]').value;
    let aux_loss_weight = document.querySelector('input[name="aux_loss_weight"]').value;

    $.ajax({
        type:'POST',
        url:'/automl/start-image-nas/',
        data:{
            "dataset_name": dataset_name,
            "layer_candidates": selected,
            "strategy": strategy,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip,
            "width": width,
            "num_of_cells": num_of_cells,
            "aux_loss_weight": aux_loss_weight
        },
        success: function(data) {
            console.log("ajax success");
        },
        error: function(xhr, errmsg, err) {
            console.log(errmsg);
        }
    })
}


// 초기 탭
showTab('image');

var ctx2 = document.getElementById("chart-line").getContext("2d");

new Chart(ctx2, {
    type: "line",
    data: {
    labels: ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
    datasets: [{
        label: "Sales",
        tension: 0,
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: "#43A047",
        pointBorderColor: "transparent",
        borderColor: "#43A047",
        backgroundColor: "transparent",
        fill: true,
        data: [120, 230, 130, 440, 250, 360, 270, 180, 90, 300, 310, 220],
        maxBarThickness: 6

    }],
    },
    options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
        display: false,
        },
        tooltip: {
        callbacks: {
            title: function(context) {
            const fullMonths = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
            return fullMonths[context[0].dataIndex];
            }
        }
        }
    },
    interaction: {
        intersect: false,
        mode: 'index',
    },
    scales: {
        y: {
        grid: {
            drawBorder: false,
            display: true,
            drawOnChartArea: true,
            drawTicks: false,
            borderDash: [4, 4],
            color: '#e5e5e5'
        },
        ticks: {
            display: true,
            color: '#737373',
            padding: 10,
            font: {
            size: 12,
            lineHeight: 2
            },
        }
        },
        x: {
        grid: {
            drawBorder: false,
            display: false,
            drawOnChartArea: false,
            drawTicks: false,
            borderDash: [5, 5]
        },
        ticks: {
            display: true,
            color: '#737373',
            padding: 10,
            font: {
            size: 12,
            lineHeight: 2
            },
        }
        },
    },
    },
});