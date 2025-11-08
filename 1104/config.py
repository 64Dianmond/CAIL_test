# 本地 vLLM OpenAI 兼容端点
OPENAI_CFG = {
    "emb_06b": {"base_url": "http://localhost:8001/v1", "api_key": "token-emb-06b",
                "model": "Qwen/Qwen3-Embedding-0.6B"},
    "emb_4b":  {"base_url": "http://192.168.201.29:8011/v1", "api_key": "token-emb-4b",
                "model": "Qwen3-Embedding-4B"},
}
# vLLM 的 rerank 端点与 Cohere 兼容（注意 base_url 不带 /v1）
RERANK_CFG = {
    "rerank_06b": {"base_url": "http://localhost:8003", "api_key": "token-rerank-06b",
                   "model": "Qwen/Qwen3-Reranker-0.6B"},
    "rerank_4b":  {"base_url": "http://192.168.201.29:8010", "api_key": "token-rerank-4b",
                   "model": "Qwen3Reranker4B"},
}
INDEX_DIR = "faiss_task2"
