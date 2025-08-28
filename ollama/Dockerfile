FROM ollama/ollama:latest

# 裝健康檢查會用到的 curl（基底映像未必有）
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# 啟動腳本
COPY init-ollama.sh /usr/local/bin/init-ollama.sh
RUN chmod +x /usr/local/bin/init-ollama.sh

ENTRYPOINT ["/usr/local/bin/init-ollama.sh"]
