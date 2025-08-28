import { store } from './storage.js';

export function appendBubble(role, content){
  const chatEl = document.getElementById('chat');
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `<div class="role">${role}</div><div class="content"></div>`;
  wrap.querySelector('.content').textContent = content;
  chatEl.appendChild(wrap);
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
