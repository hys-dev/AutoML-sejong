let charts = {};


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