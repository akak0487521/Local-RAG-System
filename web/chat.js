import { store } from './storage.js';

export function appendBubble(role, content){
  const chatEl = document.getElementById('chat');
  const wrap = document.createElement('div');
  const pending = role === 'assistant' && !content;
  wrap.className = `msg ${role}${pending ? ' pending' : ''}`;
  wrap.innerHTML = `<div class="role">${role}</div><div class="content"></div>`;
  const contentEl = wrap.querySelector('.content');
  if(pending){
    contentEl.innerHTML = '<span class="loader" aria-label="thinking"></span>';
  }else{
    contentEl.textContent = content;
  }
  chatEl.appendChild(wrap);
  return wrap;
}

export function appendReasoning(node, text){
  if(!node) return;
  let block = node._reasonBlock;
  if(!block){
    block = document.createElement('details');
    block.className = 'reasoning-block';
    const summary = document.createElement('summary');
    summary.textContent = '推理過程';
    const pre = document.createElement('pre');
    block.appendChild(summary);
    block.appendChild(pre);
    node._reasonBlock = block;
  }
  const pre = block.querySelector('pre');
  pre.textContent += text;
  if(!block.isConnected) node.appendChild(block);
}

export function renderAll(){
  const chatEl = document.getElementById('chat');
  chatEl.innerHTML = '';
  const { messages } = store.sessions[store.currentId];
  for (const m of messages){
    appendBubble(m.role, m.content);
  }
  chatEl.scrollTop = chatEl.scrollHeight;
}
