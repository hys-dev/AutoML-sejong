let charts = {};

document.getElementById('image-zip').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload-zip/', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error('업로드 실패');
    const result = await res.json();
    alert(`업로드 성공: ${result.filename}`);
  } catch (err) {
    alert(err.message);
  }
});

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
    const logEl = document.getElementById(kind==='image'?'image-log':'multi-log');
    logEl.textContent='';
    let i=0;
    const interval = setInterval(()=>{
        i++;
        logEl.textContent += `Epoch ${i}: acc=${Math.round(30+Math.random()*60)}%\n`;
        logEl.scrollTop=logEl.scrollHeight;
        if(i>=5) clearInterval(interval);
    },800);
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