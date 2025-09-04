async function loadDocs(){
  try{
    const res = await fetch('/docs/list');
    const data = await res.json();
    const tbody = document.getElementById('docTable');
    tbody.innerHTML = '';
    (data.docs || []).forEach(d => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${d.title||''}</td><td>${d.source||''}</td><td><button onclick="editDoc('${d.id}')">編輯</button> <button onclick="deleteDoc('${d.id}')">刪除</button></td>`;
      tbody.appendChild(tr);
    });
  }catch(e){ console.error('loadDocs error', e); }
}

async function uploadDoc(ev){
  ev.preventDefault();
  const title = document.getElementById('title').value.trim();
  const source = document.getElementById('source').value.trim();
  const file = document.getElementById('file').files[0];
  if(!file) return;
  const text = await file.text();
  await fetch('/docs/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ title, content:text, metadata:{ source } })
  });
  ev.target.reset();
  loadDocs();
}

async function editDoc(id){
  const title = prompt('新標題?');
  if(title===null) return;
  const source = prompt('新來源?');
  if(source===null) return;
  await fetch(`/docs/${id}`, {
    method:'PUT',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ title, source })
  });
  loadDocs();
}

async function deleteDoc(id){
  if(!confirm('確定刪除？')) return;
  await fetch(`/docs/${id}`, { method:'DELETE' });
  loadDocs();
}

document.getElementById('uploadForm').addEventListener('submit', uploadDoc);
window.editDoc = editDoc;
window.deleteDoc = deleteDoc;
loadDocs();
