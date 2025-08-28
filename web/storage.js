export const store = {
  sessions: JSON.parse(localStorage.getItem('sessions') || '{}'),
  currentId: localStorage.getItem('currentId'),
  selected: new Map(),
  ragEnabled: JSON.parse(localStorage.getItem('ragEnabled') || 'false'),
  injectWhere: localStorage.getItem('injectWhere') || 'system'
};

export function newId(){
  return 'c'+Math.random().toString(36).slice(2,10);
}

if(!store.currentId) store.currentId = newId();
if(!store.sessions[store.currentId]) {
  store.sessions[store.currentId] = { title:'新對話', messages:[] };
}

export function persist(){
  localStorage.setItem('sessions', JSON.stringify(store.sessions));
}
