export const params = {
  apiBase: localStorage.getItem('apiBase') || '',
  apiKey: localStorage.getItem('apiKey') || '',
  mode: localStorage.getItem('mode') || 'creative',
  lang: localStorage.getItem('lang') || 'zh-tw',
  engine: localStorage.getItem('engine') || 'auto',
  targetLength: localStorage.getItem('targetLength') || '',
  threadId: localStorage.getItem('threadId') || '',
  k: parseInt(localStorage.getItem('k') || '6'),
  rerank: (localStorage.getItem('rerank') || 'true') === 'true',
  namespace: localStorage.getItem('namespace') || '',
  canonicality: localStorage.getItem('canonicality') || '',
  tone: localStorage.getItem('tone') || 'neutral',
  directness: parseFloat(localStorage.getItem('directness') || '0.7'),
  empathy: parseFloat(localStorage.getItem('empathy') || '0.6'),
  hedging: parseFloat(localStorage.getItem('hedging') || '0.3'),
  formality: parseFloat(localStorage.getItem('formality') || '0.5'),
};

export function saveParams(){
  for (const [k,v] of Object.entries(params)) {
    localStorage.setItem(k, typeof v === 'boolean' ? String(v) : String(v));
  }
}
