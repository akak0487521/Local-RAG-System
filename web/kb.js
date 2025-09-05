async function loadDocs(q=''){
  try{
    const url = '/docs/list' + (q ? `?q=${encodeURIComponent(q)}` : '');
    const res = await fetch(url);
    const data = await res.json();
    const tbody = document.getElementById('docTable');
    tbody.innerHTML = '';
    const docs = data.docs || [];
    docs.forEach(d => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${d.title||''}</td><td>${d.source||''}</td><td><button onclick="editDoc('${d.id}')">編輯</button> <button onclick="deleteDoc('${d.id}')">刪除</button></td>`;
      tbody.appendChild(tr);
    });
    const treeDiv = document.getElementById('folderTree');
    treeDiv.innerHTML = '';
    const tree = buildTree(docs);
    renderTree(tree, treeDiv);
  }catch(e){ console.error('loadDocs error', e); }
}

function buildTree(docs){
  const root = {};
  docs.forEach(d => {
    const parts = (d.path || d.folder || '').split('/').filter(Boolean);
    let node = root;
    parts.forEach(p => {
      node.children = node.children || {};
      node.children[p] = node.children[p] || {};
      node = node.children[p];
    });
    node.docs = node.docs || [];
    node.docs.push(d);
  });
  return root;
}

function renderTree(node, parent){
  const ul = document.createElement('ul');
  if(node.children){
    Object.keys(node.children).sort().forEach(name => {
      const li = document.createElement('li');
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = name;
      details.appendChild(summary);
      renderTree(node.children[name], details);
      li.appendChild(details);
      ul.appendChild(li);
    });
  }
  if(node.docs){
    node.docs.forEach(d => {
      const li = document.createElement('li');
      li.innerHTML = `${d.title||''} <span class="muted">${d.source||''}</span> <button onclick="editDoc('${d.id}')">編輯</button> <button onclick="deleteDoc('${d.id}')">刪除</button>`;
      ul.appendChild(li);
    });
  }
  parent.appendChild(ul);
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
document.getElementById('searchBtn').addEventListener('click', () => {
  loadDocs(document.getElementById('searchInput').value.trim());
});
document.getElementById('searchInput').addEventListener('keydown', e => {
  if(e.key === 'Enter'){
    e.preventDefault();
    loadDocs(e.target.value.trim());
  }
});
window.editDoc = editDoc;
window.deleteDoc = deleteDoc;
loadDocs();
