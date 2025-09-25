# main.py (批量处理最终版)
import argparse
from tqdm import tqdm
import itertools
from config import (
    TASK, CORPUS_DIR, DOC_CORPUS_DIR, QRELS_DIR, TOP_K, YEAR_FILTER_RANGE, 
    CBERT_MODEL_NAME, LEGAL_BERT_MODEL_NAME, NUM_WORKERS_BM25
)
from utils import load_corpus, load_qrels
from prefetchers import BM25_Prefetcher, BERT_Prefetcher, Ensemble_Prefetcher
from evaluate import recall_at_k

def apply_time_filter(ranked_list, query_doc, doc_corpus, year_range):
    # ... (代码不变)
    query_year_str = query_doc.get("year")
    if not query_year_str: return ranked_list
    try: query_year = int(query_year_str)
    except (ValueError, TypeError): return ranked_list
    min_year, max_year = query_year - year_range['lower'], query_year + year_range['upper']
    filtered_list = []
    for doc_id, score in ranked_list:
        doc_info = doc_corpus.get(doc_id)
        if doc_info and doc_info.get("year"):
            try:
                doc_year = int(doc_info.get("year"))
                if min_year <= doc_year <= max_year:
                    filtered_list.append((doc_id, score))
            except (ValueError, TypeError): continue
    return filtered_list

def main(prefetcher_type, use_time_filter, debug_count, rebuild_index):
    # 1. 加载数据
    print("--- Loading Data ---")
    query_corpus = load_corpus(CORPUS_DIR)
    doc_corpus = load_corpus(DOC_CORPUS_DIR)
    qrels = load_qrels(QRELS_DIR)
    test_qrels = qrels['test']
    
    if debug_count:
        print(f"--- !!! DEBUG MODE: Running on first {debug_count} queries only. !!! ---")
        test_qrels = dict(itertools.islice(test_qrels.items(), debug_count))

    # 2. 初始化预取器
    print(f"\n--- Initializing {prefetcher_type.upper()} Prefetcher ---")
    if prefetcher_type == 'bm25':
        prefetcher = BM25_Prefetcher(doc_corpus)
    elif prefetcher_type == 'c-bert':
        prefetcher = BERT_Prefetcher(doc_corpus, model_path=CBERT_MODEL_NAME, rebuild_index=rebuild_index)
    elif prefetcher_type == 'legal-bert':
        prefetcher = BERT_Prefetcher(doc_corpus, model_path=LEGAL_BERT_MODEL_NAME, rebuild_index=rebuild_index)
    elif prefetcher_type == 'ensemble':
        prefetcher = Ensemble_Prefetcher(doc_corpus, cbert_path=CBERT_MODEL_NAME, rebuild_index=rebuild_index)
    else:
        raise ValueError("Prefetcher type is invalid.")

    # --- 核心修改: 从串行循环改为批量处理 ---
    print("\n--- Running Retrieval on Test Set (Batch Mode) ---")
    
    # 准备批量查询
    query_ids = list(test_qrels.keys())
    query_texts = [query_corpus[qid].get("text", "") for qid in query_ids]
    
    # 执行批量排名
    # 注意：只有Ensemble和BM25实现了rank_batch，BERT的rank本身就是批量的
    if hasattr(prefetcher, 'rank_batch'):
        all_ranked_lists = prefetcher.rank_batch(query_texts, TOP_K)
    else: # 对于纯BERT模式，我们仍然可以批量编码，但简单起见，这里保持循环
        all_ranked_lists = [prefetcher.rank(qt, TOP_K) for qt in tqdm(query_texts, desc="BERT Ranking")]
        
    # 将结果打包回字典
    results = {query_ids[i]: all_ranked_lists[i] for i in range(len(query_ids))}

    # 如果需要，应用时间过滤器
    if use_time_filter:
        print("Applying time filter...")
        for query_id in tqdm(results.keys()):
            query_doc = query_corpus[query_id]
            results[query_id] = apply_time_filter(results[query_id], query_doc, doc_corpus, YEAR_FILTER_RANGE)

    # 确保结果不超过TOP_K (时间过滤后可能会变少，但不会变多)
    for query_id in results:
        results[query_id] = results[query_id][:TOP_K]

    # --- 修改结束 ---

    # 4. 评估结果
    print("\n--- Evaluating Results ---")
    recall_10 = recall_at_k(results, test_qrels, k=10)
    recall_100 = recall_at_k(results, test_qrels, k=TOP_K)
    
    print(f"Task: {TASK.upper()}")
    print(f"Prefetcher: {prefetcher_type.upper()}")
    print(f"Time Filter Applied: {use_time_filter}")
    print(f"Number of Queries Evaluated: {len(test_qrels)}")
    print(f"Recall@10: {recall_10:.4f}")
    print(f"Recall@{TOP_K}: {recall_100:.4f}")
    
    # 清理多进程池
    if hasattr(prefetcher, 'bm25') and hasattr(prefetcher.bm25, 'stop_pool'):
        prefetcher.bm25.stop_pool()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IR experiment with optimized logic.")
    parser.add_argument('--prefetcher', type=str, default='ensemble', choices=['bm25', 'c-bert', 'legal-bert', 'ensemble'],
                        help="Choose the prefetcher model to use.")
    parser.add_argument('--time_filter', action='store_true',
                        help="Apply time-based filtering on the results.")
    parser.add_argument('--debug_count', type=int, default=None,
                        help="Run on a small subset of N queries for quick debugging.")
    parser.add_argument('--rebuild_index', action='store_true',
                        help="Force rebuild of FAISS index for the selected BERT model.")
    # --- 新增参数 ---
    parser.add_argument('--bm25_workers', type=int, default=NUM_WORKERS_BM25,
                        help="Number of parallel processes for BM25 ranking.")
    args = parser.parse_args()
    
    # 更新配置
    NUM_WORKERS_BM25 = args.bm25_workers
    
    main(args.prefetcher, args.time_filter, args.debug_count, args.rebuild_index)