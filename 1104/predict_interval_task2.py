#!/usr/bin/env python3
import os
import orjson, json, numpy as np, pandas as pd, faiss, argparse
from pathlib import Path
from openai import OpenAI
import cohere
from config import OPENAI_CFG, RERANK_CFG, INDEX_DIR
from tqdm import tqdm

# ---- 可通过环境变量调整的默认值（命令行会覆盖） ----
# 示例： export K_RECALL_DEFAULT=50; export K_DEFAULT=5
DEFAULT_K_RECALL = int(os.getenv("K_RECALL_DEFAULT", "50"))
DEFAULT_K = int(os.getenv("K_DEFAULT", "5"))

# ---- 全局索引缓存 ----
_META = None
_INDEX = None

def init_index(use_gpu=True, device=0, temp_mem_bytes=0):
    """
    只调用一次：尝试把索引拷到 GPU（受限临时内存），若失败则回退 CPU。
    返回 (meta_df, index)
    """
    global _META, _INDEX
    if _META is not None and _INDEX is not None:
        return _META, _INDEX

    meta_path = Path(INDEX_DIR) / "meta.parquet"
    idx_path = str(Path(INDEX_DIR) / "sentencing_flatip.faiss")

    if not meta_path.exists() or not Path(idx_path).exists():
        raise FileNotFoundError(f"Meta or index not found in {INDEX_DIR}")

    meta = pd.read_parquet(meta_path)
    cpu_idx = faiss.read_index(idx_path)

    # 尝试 GPU（如果可用）
    if use_gpu:
        try:
            ngpu = 0
            try:
                ngpu = faiss.get_num_gpus()
            except Exception:
                # 有些 faiss build 没有 get_num_gpus，尽量尝试一次 GPU 转移，若失败会捕获
                ngpu = 1

            if ngpu <= 0:
                raise RuntimeError("No GPUs detected by faiss.get_num_gpus()")

            # 创建资源并限制临时内存（0 表示不预分配）
            res = faiss.StandardGpuResources()
            res.setTempMemory(int(temp_mem_bytes))

            # 尝试单卡拷贝并使用 float16 以节省显存
            try:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
            except Exception:
                co = None

            gpu_idx = faiss.index_cpu_to_gpu(res, device, cpu_idx, co) if co is not None else faiss.index_cpu_to_gpu(res, device, cpu_idx)
            _META, _INDEX = meta, gpu_idx
            print("Index loaded to GPU (device {}).".format(device))
            return _META, _INDEX

        except Exception as e:
            print("GPU load failed or OOM, falling back to CPU index. Error:", e)

    # 回退到 CPU
    _META, _INDEX = meta, cpu_idx
    print("Index loaded to CPU.")
    return _META, _INDEX


# ---- Embedding & rerank helpers ----
def embed_query(text, which):
    cfg = OPENAI_CFG[which]
    cli = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    out = cli.embeddings.create(model=cfg["model"], input=[text])
    v = np.array(out.data[0].embedding, dtype="float32")[None, :]
    faiss.normalize_L2(v)
    return v

def rerank(query, docs, which="rerank_06b", top_n=50):
    """
    使用 Cohere 客户端调用 vLLM 的 /v1/rerank（兼容）
    返回列表 [(index_in_docs, score), ...]
    """
    cfg = RERANK_CFG[which]
    co = cohere.Client(api_key=cfg["api_key"], base_url=cfg["base_url"])
    rr = co.rerank(model=cfg["model"], query=query, documents=docs, top_n=min(top_n, len(docs)))
    # rr.results: each has .index (position in docs) and .relevance_score
    return [(r.index, r.relevance_score) for r in rr.results]


# ---------- 罪名识别 + 提示模板 ----------
def detect_crime_type(fact: str) -> str:
    if not fact:
        return "其他"
    s = fact
    if any(k in s for k in ["盗窃", "偷", "窃取", "入室盗窃", "扒窃", "顺手牵羊"]):
        return "盗窃"
    if any(k in s for k in ["故意伤害", "伤人", "殴打", "致伤", "重伤"]):
        return "故意伤害"
    if any(k in s for k in ["诈骗", "骗取", "诈骗罪", "虚构事实", "非法占有", "冒充"]):
        return "诈骗"
    return "其他"

TEMPLATES = {
    "盗窃": {
        "concise": "请检索与下列盗窃案件最相似的判决（优先量刑相近）。事实摘要：{fact}。关注：赃物种类与价值；是否当场/是否自首；是否退赃/赔偿；是否有前科；是否共同犯罪。",
        "element": "案件类型：盗窃罪\n事实摘要：{fact}\n关键要素：赃物（种类/数量/估价）、作案手法（入室/扒窃/顺手牵羊）、是否当场抓获、是否自首/退赃/赔偿、是否有前科、共同作案人数\n请求：按事实+量刑相似度排序并返回前 k 案例，附刑期（月）。",
        "formal": "请在候选判决中检索与下列事实要点最为接近的盗窃类案例，并优先返回量刑相近的判决。事实要点：{fact}。请返回每案的简短摘要与判决刑期（月）。"
    },
    "故意伤害": {
        "concise": "检索与下列故意伤害案件最相近的判例（优先量刑）。事实摘要：{fact}。关注：伤情程度（轻伤/重伤/致死）、是否使用凶器、是否赔偿、是否自首或认罪。",
        "element": "案件类型：故意伤害罪\n事实摘要：{fact}\n关键要素：伤情等级、作案工具、受害人数、赔偿/和解情况、是否认罪或有前科\n请求：按事实与量刑相似度排序并返回前 k 案例，附刑期（月）。",
        "formal": "依据下列事实要点检索最相近的故意伤害类判决案例，重点比较伤情级别与量刑。事实：{fact}。请返回前 k 案例的判决要点与月刑。"
    },
    "诈骗": {
        "concise": "检索与下列诈骗案最相似的判例（优先量刑）。事实摘要：{fact}。关注：诈骗金额、作案手段（电信/网络/合同）、是否退赃/赔偿、是否共同实施。",
        "element": "案件类型：诈骗罪\n事实摘要：{fact}\n关键要素：涉案金额、受害人数、诈骗手段、是否退赃/赔偿、是否认罪/前科\n请求：按事实与量刑相似度排序并返回前 k 案例，附刑期（月）。",
        "formal": "请根据下列事实要点检索与之最为接近的诈骗类判例，优先返回量刑及事实高度相似的案例。事实：{fact}。每案返回摘要与判决刑期（月）。"
    },
    "其他": {
        "concise": "请检索与下列案件事实最相似的判决（优先量刑相近）。事实摘要：{fact}。请返回前 k 案例及其刑期（月）。",
        "element": "事实摘要：{fact}\n请求：在候选文档中按事实与量刑相似度排序并返回前 k 案例及其刑期（月）。",
        "formal": "请根据以下事实检索最相近的判决案例，返回前 k 案例的摘要与判决刑期（月）。\n事实：{fact}"
    }
}

def format_rerank_query(fact: str, crime_type: str = None, style: str = "element", k: int = 5) -> str:
    """
    fact: 原始事实（你的 fact 文本）
    crime_type: "盗窃"/"故意伤害"/"诈骗"/None（None 则自动检测）
    style: "concise"/"element"/"formal"
    k: 希望返回的 top k（会放入提示里作为参考）
    """
    if crime_type is None:
        crime_type = detect_crime_type(fact)
    templates = TEMPLATES.get(crime_type, TEMPLATES["其他"])
    tpl = templates.get(style, templates["element"])
    q = tpl.format(fact=fact)
    q += f"\n\n请返回候选中的前 {k} 个最相似案例，并按相似度降序排列。"
    return q


# ---- 单条预测（返回 submit interval 和 top5 list） ----
def predict_one(fact, emb="emb_06b", rerank_model="rerank_06b", K_recall=None, k=None, rerank_style="element"):
    """
    - K_recall: 从 FAISS initial recall 的数量
    - k: 重排后要取的 top k
    返回 (interval_list, topk_list)
      interval_list: [lower_month, upper_month]  (由 top1/top2 的 months 决定)
      topk_list: list of dicts [{"meta_id":..., "fact":..., "months":..., "score":...}, ...]
    """
    # 使用默认值（优先命令行/环境变量）
    if K_recall is None:
        K_recall = DEFAULT_K_RECALL
    if k is None:
        k = DEFAULT_K

    # 校验
    if K_recall < 1:
        K_recall = 1
    if k < 1:
        k = 1
    if k > K_recall:
        # 不允许 k > K_recall，自动修正并提示
        print(f"[WARN] k ({k}) > K_recall ({K_recall}), 将 k 调整为 K_recall。")
        k = K_recall

    meta, index = init_index()

    # embed
    qv = embed_query(fact, which=emb)
    D, I = index.search(qv, K_recall)  # I shape (1, K_recall)
    raw_inds = I[0]

    # 处理 -1（如果存在）
    valid_mask = raw_inds >= 0
    valid_inds = raw_inds[valid_mask]
    if len(valid_inds) == 0:
        # 没有检索到候选，返回空结果和 [0,0]
        return [0, 0], []

    cand = meta.iloc[valid_inds].reset_index(drop=True)  # 重置位置，便于按 rerank index 访问
    docs = cand["fact"].tolist()

    # 构造针对罪名与风格的 rerank query（默认 element 风格）
    crime = detect_crime_type(fact)
    query = format_rerank_query(fact, crime_type=crime, style=rerank_style, k=k)

    # rerank top K_recall 中的 docs（只取 top k）
    try:
        pairs = rerank(query, docs, which=rerank_model, top_n=min(k, len(docs)))
    except Exception as e:
        # rerank 请求失败，回退为空并基于 cand 做简单选择
        print("Rerank failed (falling back to basic selection). Error:", e)
        pairs = []

    if not pairs:
        # rerank 未返回结果，直接基于 cand 取前两个 months
        months_list = cand["months"].astype(int).tolist()
        if len(months_list) == 0:
            return [0, 0], []
        if len(months_list) == 1:
            return [months_list[0], months_list[0]], [{"meta_id": int(cand.iloc[0]["id"]) if "id" in cand.columns else None, "fact": cand.iloc[0]["fact"], "months": int(months_list[0]), "score": None}]
        # 两个或更多
        lower = min(int(months_list[0]), int(months_list[1])); upper = max(int(months_list[0]), int(months_list[1]))
        topk_list = []
        for i in range(min(5, len(months_list))):
            topk_list.append({"meta_id": int(cand.iloc[i]["id"]) if "id" in cand.columns else None, "fact": cand.iloc[i]["fact"], "months": int(months_list[i]), "score": None})
        return [lower, upper], topk_list

    # pairs 是 (idx_in_docs, score)
    pairs_sorted = sorted(pairs, key=lambda x: -x[1])
    topk_pairs = pairs_sorted[:k]

    topk_list = []
    for idx_in_docs, score in topk_pairs[:5]:  # 只保存 top5
        row = cand.iloc[int(idx_in_docs)]
        topk_list.append({
            "meta_id": int(row["id"]) if "id" in row else None,
            "fact": row["fact"],
            "months": int(row["months"]),
            "score": float(score)
        })

    # 构造 submit interval：只取 top1 和 top2 的 months（若不足则复制）
    if len(topk_list) == 0:
        interval = [0, 0]
    elif len(topk_list) == 1:
        m = topk_list[0]["months"]
        interval = [m, m]
    else:
        m1 = topk_list[0]["months"]; m2 = topk_list[1]["months"]
        lower = int(min(m1, m2)); upper = int(max(m1, m2))
        interval = [lower, upper]

    return interval, topk_list


# ---- 批处理：对每行生成 submit 与 top5 两个 jsonl 输出（带进度条） ----
def predict_file(in_path, submit_out, top5_out,
                 emb="emb_06b", rerank_model="rerank_06b",
                 K_recall=None, k=None, use_gpu=True, gpu_device=0, rerank_style="element"):
    # 初始化索引（只加载一次）
    init_index(use_gpu=use_gpu, device=gpu_device, temp_mem_bytes=0)

    # 计算总行数以用于 tqdm
    with open(in_path, "rb") as f:
        total = sum(1 for _ in f)

    # 传入的 K_recall/k 可为 None（此时 predict_one 使用默认）
    with open(in_path, "rb") as fin, open(submit_out, "wb") as fout_sub, open(top5_out, "wb") as fout_top5:
        for raw in tqdm(fin, total=total, desc="Processing"):
            if not raw.strip():
                continue
            obj = orjson.loads(raw)
            id_ = obj.get("id")
            fact = obj.get("fact", "")

            interval, topk_list = predict_one(fact, emb=emb, rerank_model=rerank_model, K_recall=K_recall, k=k, rerank_style=rerank_style)

            # 写 submit（只包含 top1/top2 构成的区间）
            rec = {"id": id_, "answer2": interval}
            fout_sub.write(orjson.dumps(rec, option=orjson.OPT_APPEND_NEWLINE))

            # 写 top5 文件：保存 top5 的 fact + months + score + meta_id
            rec2 = {"id": id_, "top5": topk_list}
            fout_top5.write(orjson.dumps(rec2, option=orjson.OPT_APPEND_NEWLINE))


# ---- CLI ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=str, help="单条案件 fact 文本")
    ap.add_argument("--in_file", type=str, help="task jsonl（含 id/fact）")
    ap.add_argument("--out_file", type=str, default="submit_task2.jsonl", help="提交文件")
    ap.add_argument("--top5_out", type=str, default="top5_cases.jsonl", help="每条 id 的 top5 案例输出文件")
    ap.add_argument("--emb", type=str, default="emb_4b", choices=["emb_06b", "emb_4b"])
    ap.add_argument("--rerank", type=str, default="rerank_4b", choices=["rerank_06b", "rerank_4b"])
    ap.add_argument("--K_recall", type=int, default=DEFAULT_K_RECALL, help=f"FAISS recall count (默认 {DEFAULT_K_RECALL})")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help=f"rerank top k (默认 {DEFAULT_K})")
    ap.add_argument("--use_gpu", action="store_true", help="尝试把索引载入 GPU（失败会回退 CPU）")
    ap.add_argument("--gpu_device", type=int, default=0)
    ap.add_argument("--rerank_style", type=str, default="element", choices=["concise", "element", "formal"], help="rerank prompt 风格")
    args = ap.parse_args()

    if args.single:
        interval, topk = predict_one(args.single, emb=args.emb, rerank_model=args.rerank,
                                     K_recall=args.K_recall, k=args.k, rerank_style=args.rerank_style)
        print("预测刑期区间（月）:", interval)
        print("Topk (<=5):")
        for t in topk:
            print(t)
    elif args.in_file:
        predict_file(args.in_file, args.out_file, args.top5_out,
                     emb=args.emb, rerank_model=args.rerank,
                     K_recall=args.K_recall, k=args.k,
                     use_gpu=args.use_gpu, gpu_device=args.gpu_device,
                     rerank_style=args.rerank_style)
        print("已生成：", args.out_file, "和", args.top5_out)
    else:
        ap.error("必须提供 --single 或 --in_file")
