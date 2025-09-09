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
      const titleTd = document.createElement('td');
      titleTd.textContent = d.title || '';
      const fileTd = document.createElement('td');
      fileTd.textContent = d.file || d.source || '';
      const actionsTd = document.createElement('td');
      const editBtn = document.createElement('button');
      editBtn.textContent = '編輯';
      editBtn.addEventListener('click', function(){ openEditModal(d.id); });
      const delBtn = document.createElement('button');
      delBtn.textContent = '刪除';
      delBtn.addEventListener('click', function(){ deleteDoc(d.id); });
      actionsTd.append(editBtn, ' ', delBtn);
      tr.append(titleTd, fileTd, actionsTd);
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
      const titleSpan = document.createElement('span');
      titleSpan.textContent = d.title || '';
      const fileSpan = document.createElement('span');
      fileSpan.className = 'muted';
      fileSpan.textContent = d.file || d.source || '';
      const editBtn = document.createElement('button');
      editBtn.textContent = '編輯';
      editBtn.addEventListener('click', function(){ openEditModal(d.id); });
      const delBtn = document.createElement('button');
      delBtn.textContent = '刪除';
      delBtn.addEventListener('click', function(){ deleteDoc(d.id); });
      li.append(titleSpan, ' ', fileSpan, ' ', editBtn, ' ', delBtn);
      ul.appendChild(li);
    });
  }
  parent.appendChild(ul);
}

function renderBodyEditor(value){
  const container = document.getElementById('docBodyEditor');
  container.innerHTML = '';
  container.appendChild(renderValue(value));
}

function renderValue(value){
  if(Array.isArray(value)) return renderArray(value);
  if(value && typeof value === 'object') return renderObject(value);
  return renderPrimitive(value);
}

function renderObject(obj){
  const div = document.createElement('div');
  div.className = 'tree-object';
  div.dataset.type = 'object';
  Object.entries(obj).forEach(([k,v]) => {
    div.appendChild(renderObjectRow(k,v));
  });
  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.textContent = '增加下層';
  addBtn.className = 'tree-add-child';
  addBtn.addEventListener('click', e => {
    e.preventDefault();
    e.stopPropagation();
    div.insertBefore(renderObjectRow('', ''), addBtn);
  });
  div.appendChild(addBtn);
  return div;
}

function renderObjectRow(key, value){
  const row = document.createElement('div');
  row.className = 'tree-row';
  const keyLabel = document.createElement('textarea');
  keyLabel.className = 'tree-label';
  keyLabel.value = key;
  keyLabel.rows = 1;
  makeEditable(keyLabel);
  const valDiv = document.createElement('div');
  valDiv.className = 'tree-value';
  valDiv.appendChild(renderValue(value));
  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.textContent = '增加下層';
  addBtn.className = 'tree-add-child';
  addBtn.addEventListener('click', e => {
    e.preventDefault();
    e.stopPropagation();
    const container = e.currentTarget.previousElementSibling;
    addChild(container);
  });
  const rmBtn = document.createElement('button');
  rmBtn.type = 'button';
  rmBtn.textContent = '刪除';
  rmBtn.className = 'tree-remove';
  rmBtn.addEventListener('click', e => {
    e.stopPropagation();
    row.remove();
  });
  row.append(keyLabel, valDiv, addBtn, rmBtn);
  return row;
}

function renderArray(arr){
  const div = document.createElement('div');
  div.className = 'tree-array';
  div.dataset.type = 'array';
  arr.forEach(v => div.appendChild(renderArrayRow(v)));
  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.textContent = '增加下層';
  addBtn.className = 'tree-add-child';
  addBtn.addEventListener('click', e => {
    e.preventDefault();
    e.stopPropagation();
    div.insertBefore(renderArrayRow(''), addBtn);
  });
  div.appendChild(addBtn);
  return div;
}

function renderArrayRow(value){
  const row = document.createElement('div');
  row.className = 'tree-row';
  const valDiv = document.createElement('div');
  valDiv.className = 'tree-value';
  valDiv.appendChild(renderValue(value));
  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.textContent = '增加下層';
  addBtn.className = 'tree-add-child';
  addBtn.addEventListener('click', e => {
    e.preventDefault();
    e.stopPropagation();
    const container = e.currentTarget.previousElementSibling;
    addChild(container);
  });
  const rmBtn = document.createElement('button');
  rmBtn.type = 'button';
  rmBtn.textContent = '刪除';
  rmBtn.className = 'tree-remove';
  rmBtn.addEventListener('click', e => {
    e.stopPropagation();
    row.remove();
  });
  row.append(valDiv, addBtn, rmBtn);
  return row;
}

function renderPrimitive(value){
  const ta = document.createElement('textarea');
  ta.className = 'tree-label';
  ta.dataset.type = 'primitive';
  ta.value = value;
  ta.rows = 1;
  makeEditable(ta);
  return ta;
}

function makeEditable(el){
  if(!(el instanceof HTMLTextAreaElement)){
    el.contentEditable = true;
  }
  el.addEventListener('click', e => {
    e.stopPropagation();
  });
  el.addEventListener('keydown', e => {
    e.stopPropagation();
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      el.blur();
    }
  });

function makeEditable(el){
  el.contentEditable = true;
  el.addEventListener('click', e => {
    e.stopPropagation();
  });
  el.addEventListener('keydown', e => {
    e.stopPropagation();
    if(e.key === 'Enter'){
      e.preventDefault();
      el.blur();
    }
  });
}

function addChild(container){
  const node = container.firstElementChild;
  if(!node){
    const obj = renderObject({});
    container.appendChild(obj);
    obj.insertBefore(renderObjectRow('', ''), obj.lastElementChild);
    return;
  }
  if(node.dataset.type === 'object'){
    node.insertBefore(renderObjectRow('', ''), node.lastElementChild);
  }else if(node.dataset.type === 'array'){
    node.insertBefore(renderArrayRow(''), node.lastElementChild);
  }else{
    const obj = renderObject({});
    container.innerHTML = '';
    container.appendChild(obj);
    obj.insertBefore(renderObjectRow('', ''), obj.lastElementChild);
  }
}

function editorToJson(){
  const container = document.getElementById('docBodyEditor');
  if(!container.firstElementChild) return {};
  return readValue(container.firstElementChild);
}

function readValue(node){
  const type = node.dataset.type;
  if(type === 'object'){
    const obj = {};
    Array.from(node.children).forEach(ch => {
      if(!ch.classList.contains('tree-row')) return;
      const keyEl = ch.querySelector('.tree-label');
      const key = keyEl ? (keyEl.tagName === 'TEXTAREA' ? keyEl.value.trim() : keyEl.textContent.trim()) : '';
      const valNode = ch.querySelector('.tree-value').firstElementChild;
      if(key) obj[key] = readValue(valNode);
    });
    return obj;
  }else if(type === 'array'){
    const arr = [];
    Array.from(node.children).forEach(ch => {
      if(!ch.classList.contains('tree-row')) return;
      const valNode = ch.querySelector('.tree-value').firstElementChild;
      arr.push(readValue(valNode));
    });
    return arr;
  }else{
    const val = node.tagName === 'TEXTAREA' ? node.value.trim() : node.textContent.trim();
    if(val === '') return '';
    try{ return JSON.parse(val); }
    catch{ return val; }
  }
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
    let doc = data.content ? JSON.parse(data.content) : data;
    const meta = data.metadata || {};
    doc.id = doc.id || data.id || meta.id || '';
    doc.namespace = doc.namespace || data.namespace || meta.namespace || '';
    doc.type = doc.type || data.type || meta.type || '';
    doc.title = doc.title || data.title || meta.title || '';
    doc.summary = doc.summary || data.summary || meta.summary || '';
    doc.body = doc.body || data.body || meta.body || '';
    doc.tags = doc.tags || data.tags || meta.tags || [];
    doc.canonicality = doc.canonicality || data.canonicality || meta.canonicality || '';
    doc.version = doc.version || data.version || meta.version || '';
    document.getElementById('docId').value = doc.id;
    document.getElementById('docNamespace').value = doc.namespace;
    document.getElementById('docType').value = doc.type;
    document.getElementById('docTitle').value = doc.title;
    document.getElementById('docSummary').value = doc.summary;
    let bodyData;
    if(typeof doc.body === 'object'){ bodyData = doc.body; }
    else{ try{ bodyData = JSON.parse(doc.body||'{}'); }catch{ bodyData = {}; } }
    renderBodyEditor(bodyData);
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
  const body = editorToJson();
  const tags = document.getElementById('docTags').value.split(',').map(t=>t.trim()).filter(Boolean);
  const canonicality = document.getElementById('docCanonicality').value.trim();
  const version = document.getElementById('docVersion').value.trim();
  const payload = { id, namespace, type, title, summary, body, tags, canonicality, version };
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
