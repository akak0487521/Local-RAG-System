import { params } from "./params.js";

function buildHeaders(extra={}){ const h = { ...extra }; if(params.apiKey) h['x-api-key'] = params.apiKey; return h; }

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
      tr.innerHTML = `<td>${d.title||''}</td><td>${d.file||d.source||''}</td><td><button onclick="openEditModal('${d.id}')">編輯</button> <button onclick="deleteDoc('${d.id}')">刪除</button></td>`;
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
    const parts = (d.folder || d.path || '').split('/').filter(Boolean);
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
      li.innerHTML = `${d.title||''} <span class="muted">${d.file||d.source||''}</span> <button onclick="openEditModal('${d.id}')">編輯</button> <button onclick="deleteDoc('${d.id}')">刪除</button>`;
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

  const payload = { title, content:text };
  if(source) payload.metadata = { source };
  await fetch('/docs/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  ev.target.reset();
  loadDocs();
}

async function openEditModal(id){
  try{
    const res = await fetch(`/docs/${id}`, { headers: buildHeaders() });
    if(!res.ok){ alert('取得文件失敗'); return; }
    const data = await res.json();
    let doc;
    try{ doc = JSON.parse(data.content||'{}'); }
    catch{ doc = { id:data.id, title:data.title, body:data.content, ...(data.metadata||{}) }; }
    doc.id = doc.id || data.id || '';
    doc.namespace = doc.namespace || (data.metadata||{}).namespace || '';
    doc.type = doc.type || (data.metadata||{}).type || '';
    doc.title = doc.title || data.title || '';
    doc.summary = doc.summary || (data.metadata||{}).summary || '';
    doc.body = doc.body || data.content || '';
    doc.tags = doc.tags || (data.metadata||{}).tags || [];
    doc.canonicality = doc.canonicality || (data.metadata||{}).canonicality || '';
    doc.version = doc.version || (data.metadata||{}).version || '';
    document.getElementById('docId').value = doc.id;
    document.getElementById('docNamespace').value = doc.namespace;
    document.getElementById('docType').value = doc.type;
    document.getElementById('docTitle').value = doc.title;
    document.getElementById('docSummary').value = doc.summary;
    const bodyField = document.getElementById('docBody');
    if(typeof doc.body === 'object'){ bodyField.value = JSON.stringify(doc.body, null, 2); }
    else{ bodyField.value = doc.body; }
    document.getElementById('docTags').value = (doc.tags || []).join(', ');
    document.getElementById('docCanonicality').value = doc.canonicality;
    document.getElementById('docVersion').value = doc.version;
    document.getElementById('docModal').classList.add('show');
  }catch(e){ console.error('openEditModal error', e); alert('載入文件發生錯誤'); }
}

function hideDocModal(){
  document.getElementById('docModal').classList.remove('show');
}

async function saveDoc(){
  const id = document.getElementById('docId').value.trim();
  const namespace = document.getElementById('docNamespace').value.trim();
  const type = document.getElementById('docType').value.trim();
  const title = document.getElementById('docTitle').value.trim();
  const summary = document.getElementById('docSummary').value.trim();
  const bodyText = document.getElementById('docBody').value.trim();
  let body;
  try{ body = bodyText ? JSON.parse(bodyText) : {}; }
  catch(e){ alert('Body JSON 格式錯誤'); return; }
  const tags = document.getElementById('docTags').value.split(',').map(t=>t.trim()).filter(Boolean);
  const canonicality = document.getElementById('docCanonicality').value.trim();
  const version = document.getElementById('docVersion').value.trim();
  const doc = { id, namespace, type, title, summary, body, tags, canonicality, version };
  const payload = {
    title,
    content: JSON.stringify(doc, null, 2),
    metadata: { namespace, type, summary, tags, canonicality, version }
  };
  try{
    const res = await fetch(`/docs/${id}`, {
      method:'PUT',
      headers:{ 'Content-Type':'application/json', ...buildHeaders() },
      body: JSON.stringify(payload)
    });
    if(!res.ok){ alert('儲存失敗'); return; }
    hideDocModal();
    loadDocs();
  }catch(e){ console.error('saveDoc error', e); alert('儲存時發生錯誤'); }
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
window.openEditModal = openEditModal;
window.saveDoc = saveDoc;
window.hideDocModal = hideDocModal;
window.deleteDoc = deleteDoc;
loadDocs();
