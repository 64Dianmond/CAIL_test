import os, glob, orjson, pandas as pd, numpy as np, faiss
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from config import OPENAI_CFG, INDEX_DIR


def embed_texts(texts, which="emb_06b", batch=20, max_tokens_per_batch=30000, skip_long_text=True, max_length=28000):
    """
    固定批次大小为 20 条文本
    batch: 固定每批发送的文本数量
    max_tokens_per_batch: 安全阈值，仅用于警告
    skip_long_text: 是否跳过超长文本
    max_length: 单条文本最大字符数（估算约30000 tokens）
    """
    cfg = OPENAI_CFG[which]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

    # 简单的token估算：1个中文字符≈1.5 tokens
    def estimate_tokens(text):
        return int(len(text) * 1.5)

    # 预初始化向量列表，用 None 占位
    vecs = [None] * len(texts)
    skipped_indices = []  # 记录被跳过的索引

    print(f"\n开始生成embeddings (模型: {cfg['model']})")
    print(f"配置: 固定batch={batch}条/次, 跳过超长文本={skip_long_text}, 最大字符数={max_length}")
    pbar = tqdm(total=len(texts), desc="Embedding进度", unit="条")

    # 按固定大小分批
    for i in range(0, len(texts), batch):
        current_batch_texts = texts[i:i + batch]
        batch_indices = list(range(i, min(i + batch, len(texts))))

        # 过滤超长文本
        filtered_batch = []
        filtered_indices = []

        if skip_long_text:
            for idx, text in zip(batch_indices, current_batch_texts):
                text_len = len(text)
                estimated_tokens = estimate_tokens(text)

                # 如果单条文本估算就超过 30000 tokens，跳过
                if text_len > max_length or estimated_tokens > 30000:
                    print(f"\n跳过第{idx}条超长文本: 字符数={text_len}, 估算tokens={estimated_tokens}")
                    skipped_indices.append(idx)
                    vecs[idx] = None  # 保持 None 占位
                else:
                    filtered_batch.append(text)
                    filtered_indices.append(idx)
        else:
            filtered_batch = current_batch_texts
            filtered_indices = batch_indices

        # 如果整批都被过滤了
        if not filtered_batch:
            pbar.update(len(batch_indices))
            continue

        # 估算当前批次的token数（仅用于监控）
        current_tokens = sum(estimate_tokens(text) for text in filtered_batch)

        # 如果超过阈值，发出警告但仍然尝试发送
        if current_tokens > max_tokens_per_batch:
            print(f"\n警告: 第{i // batch + 1}批次估算token数({current_tokens})超过阈值({max_tokens_per_batch})")
            print(f"  批次包含 {len(filtered_batch)} 条文本")

        try:
            res = client.embeddings.create(model=cfg["model"], input=filtered_batch)

            # 将embeddings分配到正确的位置
            embeddings = [d.embedding for d in res.data]
            for idx, emb in zip(filtered_indices, embeddings):
                vecs[idx] = emb

            pbar.update(len(filtered_batch))

        except Exception as e:
            print(f"\n错误: 第{i // batch + 1}批次API调用失败")
            print(f"  批次大小: {len(filtered_batch)}")
            print(f"  估算token数: {current_tokens}")
            print(f"  错误信息: {str(e)}")

            # 如果是 token 超限错误，跳过这批数据
            if "maximum context length" in str(e):
                print(f"  由于token超限，跳过这批数据")
                for idx in filtered_indices:
                    if idx not in skipped_indices:
                        skipped_indices.append(idx)
                        vecs[idx] = None
                pbar.update(len(filtered_indices))
                continue
            else:
                raise

    pbar.close()

    # 处理跳过的文本：用零向量替代
    if skipped_indices:
        print(f"\n共跳过 {len(skipped_indices)} 条超长文本")
        # 获取向量维度
        valid_vecs = [v for v in vecs if v is not None]
        if valid_vecs:
            dim = len(valid_vecs[0])
            # 替换 None 为零向量
            vecs = [v if v is not None else [0.0] * dim for v in vecs]
        else:
            print("错误: 所有文本都被跳过了！")
            return None, []

    X = np.asarray(vecs, dtype="float32")
    print(f"✓ Embeddings生成完成: shape={X.shape}")

    print("正在归一化向量...")
    faiss.normalize_L2(X)
    print("✓ 归一化完成")

    return X, skipped_indices


def load_task_like(paths):
    rows = []
    for p in paths:
        if p.endswith(".jsonl"):
            with open(p, "rb") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = orjson.loads(line)
                    rows.append(obj)
        else:
            with open(p, "rb") as f:
                # 允许是 list[dict] 或按行的 jsonl
                txt = f.read().decode("utf-8")
                try:
                    data = orjson.loads(txt)
                    if isinstance(data, list):
                        rows += data
                    else:
                        rows.append(data)
                except:
                    for line in txt.splitlines():
                        if line.strip():
                            rows.append(orjson.loads(line))
    return rows


def main(data_glob, emb="emb_06b", batch_size=20):
    """
    默认每批固定 20 条文本
    """
    print(f"\n{'=' * 70}")
    print(f"刑期预测索引构建程序")
    print(f"{'=' * 70}\n")

    files = glob.glob(data_glob)
    print(f"找到 {len(files)} 个数据文件:")
    for f in files:
        print(f"  - {f}")

    os.makedirs(INDEX_DIR, exist_ok=True)
    meta_path = Path(INDEX_DIR) / "meta.parquet"
    idx_path = Path(INDEX_DIR) / "sentencing_flatip.faiss"

    # 训练数据需要包含 meta.term_of_imprisonment.imprisonment（月）
    print(f"\n第1步: 加载和过滤数据...")
    facts, months, metas = [], [], []

    all_data = load_task_like(files)
    print(f"共加载 {len(all_data)} 条原始数据")

    # 统计过滤情况
    filtered_stats = {
        "no_fact": 0,
        "no_imprisonment": 0,
        "life_imprisonment": 0,
        "death_penalty": 0,
        "valid": 0
    }

    for obj in tqdm(all_data, desc="过滤数据", unit="条"):
        fact = (obj.get("fact") or "").strip()
        meta = obj.get("meta") or {}
        term = meta.get("term_of_imprisonment") or {}
        imp = term.get("imprisonment")
        life = term.get("life_imprisonment", False)
        death = term.get("death_penalty", False)

        if not fact:
            filtered_stats["no_fact"] += 1
            continue
        if imp is None:
            filtered_stats["no_imprisonment"] += 1
            continue
        if life:
            filtered_stats["life_imprisonment"] += 1
            continue
        if death:
            filtered_stats["death_penalty"] += 1
            continue

        filtered_stats["valid"] += 1
        facts.append(fact)
        months.append(int(imp))
        metas.append({"id": len(metas), "fact": fact, "months": int(imp)})

    print(f"\n数据过滤统计:")
    print(f"  ✓ 有效数据: {filtered_stats['valid']}")
    print(f"  ✗ 缺少案情描述: {filtered_stats['no_fact']}")
    print(f"  ✗ 缺少刑期信息: {filtered_stats['no_imprisonment']}")
    print(f"  ✗ 无期徒刑: {filtered_stats['life_imprisonment']}")
    print(f"  ✗ 死刑: {filtered_stats['death_penalty']}")

    if len(facts) == 0:
        print("\n错误: 没有有效数据可以建索引!")
        return

    # 刑期分布统计
    months_array = np.array(months)
    print(f"\n刑期分布统计 (单位: 月):")
    print(f"  最小值: {months_array.min()}")
    print(f"  最大值: {months_array.max()}")
    print(f"  平均值: {months_array.mean():.2f}")
    print(f"  中位数: {np.median(months_array):.2f}")

    print(f"\n第2步: 生成向量embeddings...")
    result = embed_texts(facts, which=emb, batch=batch_size, skip_long_text=True, max_length=28000)

    if result is None:
        print("\n错误: 无法生成embeddings!")
        return

    X, skipped_indices = result

    # 从 metas 中移除被跳过的数据
    if skipped_indices:
        print(f"\n移除 {len(skipped_indices)} 条被跳过的数据...")
        # 创建保留的索引掩码
        keep_mask = np.ones(len(facts), dtype=bool)
        keep_mask[skipped_indices] = False

        # 过滤数据
        X = X[keep_mask]
        metas = [m for i, m in enumerate(metas) if i not in skipped_indices]
        facts = [f for i, f in enumerate(facts) if i not in skipped_indices]
        months = [m for i, m in enumerate(months) if i not in skipped_indices]

        # 重新生成 id
        for i, m in enumerate(metas):
            m["id"] = i

        print(f"✓ 剩余有效数据: {len(metas)} 条")

    # 构建Faiss索引
    print(f"\n第3步: 构建Faiss索引...")
    d = X.shape[1]
    cpu = faiss.IndexFlatIP(d)

    print("正在将索引转移到GPU...")
    gpu = faiss.index_cpu_to_all_gpus(cpu)

    print(f"正在添加 {len(X)} 个向量到索引...")
    gpu.add(X)
    print("✓ 向量添加完成")

    print("正在将索引从GPU转回CPU并保存...")
    faiss.write_index(faiss.index_gpu_to_cpu(gpu), str(idx_path))
    print(f"✓ Faiss索引已保存: {idx_path}")

    # 保存元数据
    print(f"\n第4步: 保存元数据...")
    pd.DataFrame(metas).to_parquet(meta_path, index=False)
    print(f"✓ 元数据已保存: {meta_path}")

    # 最终总结
    print(f"\n{'=' * 70}")
    print(f"索引构建完成!")
    print(f"{'=' * 70}")
    print(f"索引文件: {idx_path}")
    print(f"元数据文件: {meta_path}")
    print(f"总向量数: {len(metas)}")
    print(f"向量维度: {d}")
    print(f"Embedding模型: {emb}")
    print(f"批次大小: {batch_size}条/次")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python build_index_for_task2.py <数据文件glob模式> [embedding模型] [batch大小]")
        print("示例: python build_index_for_task2.py 'exercise_contest/filtered_cases.jsonl' emb_4b 20")
        sys.exit(1)

    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    main(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else "emb_06b",
        batch_size
    )

