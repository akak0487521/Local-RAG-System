#!/usr/bin/env bash
set -euo pipefail

# 1) 先啟動 ollama 後端
ollama serve &

# 2) 等 API 起來
until curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; do
  sleep 0.3
done

# 3) 若尚未建立自訂模型，則拉底模並建立一次
if ! ollama list | awk '{print $1}' | grep -qx "llama3-8b-8k"; then
  ollama pull llama3:8b
  ollama create llama3-8b-8k -f /modelfiles/llama3-8b-8k.Modelfile
fi

# 4) 以前景方式等待（保持容器存活）
wait -n
