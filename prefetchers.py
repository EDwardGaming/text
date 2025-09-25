# prefetchers.py (Linux多进程加速版)
import os
import json
import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from config import (
    DEVICE, BM25_K1, BM25_B, ENSEMBLE_ALPHA, get_index_paths,
    BM25_TOPN, NUM_WORKERS_BM25, CBERT_MODEL_NAME  # 导入新的性能配置
)
# 导入多进程库
from multiprocessing import Pool, Manager
import functools

# --- 全局变量，用于在多进程中共享模型 ---
# 我们不能直接传递大的self.bm25对象，但可以共享初始化所需的数据
_bm25_tokenized_corpus = None
_bm25_doc_ids = None

def init_bm25_worker(tokenized_corpus, doc_ids):
    """初始化每个工作进程，加载BM25模型。"""
    global _bm25_tokenized_corpus, _bm25_doc_ids
    _bm25_tokenized_corpus = tokenized_corpus
    _bm25_doc_ids = doc_ids

def rank_bm25_query_worker(query_text):
    """单个工作进程执行BM25查询的函数。"""
    global _bm25_tokenized_corpus, _bm25_doc_ids
    # 在worker内部实例化一个临时的BM25对象
    bm25 = BM25Okapi(_bm25_tokenized_corpus, k1=BM25_K1, b=BM25_B)
    
    tokenized_query = query_text.split()
    top_n_ids = bm25.get_top_n(tokenized_query, _bm25_doc_ids, n=BM25_TOPN)
    
    # 返回ID列表即可，分数可以在主进程中需要时再计算，或者在这里计算并返回
    # 为了效率，我们只返回ID
    return top_n_ids

class BM25_Prefetcher:
    """BM25 预取器 (支持多进程)"""
    def __init__(self, doc_corpus):
        self.doc_corpus = doc_corpus
        self.doc_ids = list(doc_corpus.keys())
        print(f"Tokenizing corpus for BM25 with k1={BM25_K1}, b={BM25_B}...")
        self.tokenized_corpus = [doc.get("text", "").split() for doc in tqdm(doc_corpus.values())]
        
        # 单进程模式下使用的BM25实例
        self.bm25_single = BM25Okapi(self.tokenized_corpus, k1=BM25_K1, b=BM25_B)
        
        # 多进程池
        self.pool = None

    def start_pool(self):
        """启动多进程池"""
        if NUM_WORKERS_BM25 > 0:
            print(f"Starting BM25 multiprocessing pool with {NUM_WORKERS_BM25} workers...")
            # 使用forkserver启动模式更稳定
            # ctx = multiprocessing.get_context('forkserver')
            # self.pool = ctx.Pool(
            self.pool = Pool(
                processes=NUM_WORKERS_BM25,
                initializer=init_bm25_worker,
                initargs=(self.tokenized_corpus, self.doc_ids)
            )

    def rank(self, query_text, top_k):
        """单查询排名"""
        tokenized_query = query_text.split()
        top_n_ids = self.bm25_single.get_top_n(tokenized_query, self.doc_ids, n=top_k)
        all_scores = self.bm25_single.get_scores(tokenized_query)
        id_to_score_map = {self.doc_ids[i]: all_scores[i] for i in range(len(self.doc_ids))}
        return [(doc_id, id_to_score_map[doc_id]) for doc_id in top_n_ids]

    def rank_batch(self, query_texts):
        """批量查询排名 (使用多进程)"""
        if self.pool is None:
            self.start_pool()
        
        # 使用进程池并行处理所有查询
        results_ids = list(tqdm(self.pool.imap(rank_bm25_query_worker, query_texts), total=len(query_texts), desc="BM25 Batch Ranking"))
        return results_ids

    def stop_pool(self):
        """关闭多进程池"""
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
            print("BM25 multiprocessing pool stopped.")

# BERT_Prefetcher 类保持不变
class BERT_Prefetcher:
    """BERT 预取器 (GPU 加速版)"""
    def __init__(self, doc_corpus, model_path, rebuild_index=False):
        self.doc_corpus = doc_corpus
        self.model_path = model_path
        self.index_paths = get_index_paths(model_path)
        
        print(f"Loading BERT model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(DEVICE)
        self.model.eval()

        if rebuild_index and os.path.exists(self.index_paths["faiss_index"]):
            print("Rebuilding index as requested...")
            for path in self.index_paths.values():
                if os.path.exists(path): os.remove(path)
        
        if not os.path.exists(self.index_paths["faiss_index"]):
            print("Creating new FAISS index for this model...")
            self._create_index()
        else:
            print(f"Loading existing FAISS index from: {self.index_paths['faiss_index']}")
            cpu_index = faiss.read_index(self.index_paths["faiss_index"])
            if faiss.get_num_gpus() > 0:
                print(f"Found {faiss.get_num_gpus()} GPUs. Moving index to GPU 0...")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.index = cpu_index
            with open(self.index_paths["doc_ids"], 'r') as f:
                self.doc_ids = json.load(f)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed_text(self, texts, batch_size=128):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
                model_output = self.model(**encoded_input)
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            all_embeddings.append(sentence_embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def _create_index(self):
        self.doc_ids = list(self.doc_corpus.keys())
        corpus_texts = [doc.get("text", "") for doc in self.doc_corpus.values()]
        print("Generating embeddings for the document corpus...")
        all_embeddings = self._embed_text(corpus_texts)
        np.save(self.index_paths["embeddings"], all_embeddings)
        with open(self.index_paths["doc_ids"], 'w') as f:
            json.dump(self.doc_ids, f)
        embedding_dim = all_embeddings.shape[1]
        cpu_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(all_embeddings)
        cpu_index.add(all_embeddings)
        print(f"CPU Index created with {cpu_index.ntotal} vectors.")
        faiss.write_index(cpu_index, self.index_paths["faiss_index"])
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else: self.index = cpu_index

    def rank(self, query_text, top_k):
        query_embedding = self._embed_text([query_text])
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.doc_ids[i], float(distances[0][j])) for j, i in enumerate(indices[0])]

    def rerank_batch(self, query_texts, list_of_doc_ids):
        """对一批查询和它们各自的候选文档列表进行批量重排"""
        results = []
        # 将查询和其候选文档配对
        tasks = []
        for i, doc_ids in enumerate(list_of_doc_ids):
            if doc_ids:
                query_text = query_texts[i]
                docs_text = [self.doc_corpus[doc_id].get("text", "") for doc_id in doc_ids]
                tasks.append((query_text, docs_text, doc_ids))
        
        # 批量编码所有需要的文本
        all_query_texts_flat = [t[0] for t in tasks]
        all_docs_texts_flat = [text for t in tasks for text in t[1]]
        
        print("Embedding queries and docs for reranking...")
        query_embeddings = self._embed_text(all_query_texts_flat)
        doc_embeddings = self._embed_text(all_docs_texts_flat)
        
        # 重新组合并计算分数
        doc_emb_idx = 0
        for i in range(len(tasks)):
            num_docs = len(tasks[i][1])
            query_emb = query_embeddings[i:i+1]
            doc_embs = doc_embeddings[doc_emb_idx : doc_emb_idx + num_docs]
            
            scores = (query_emb @ doc_embs.T)[0]
            
            reranked_results = sorted(zip(tasks[i][2], scores), key=lambda x: x[1], reverse=True)
            results.append(reranked_results)
            
            doc_emb_idx += num_docs
            
        return results

class Ensemble_Prefetcher:
    """集成 BM25 和 BERT 的预取器 (批量+多进程优化版)"""
    def __init__(self, doc_corpus, cbert_path, rebuild_index=False):
        print("Initializing BM25 for Ensemble...")
        self.bm25 = BM25_Prefetcher(doc_corpus)
        print("\nInitializing C-BERT for Ensemble...")
        self.cbert = BERT_Prefetcher(doc_corpus, model_path=cbert_path, rebuild_index=rebuild_index)
        
    def rank_batch(self, query_texts, top_k):
        # 步骤 1: BM25 并行批量召回
        print("Step 1: BM25 is retrieving candidates in parallel...")
        bm25_candidate_ids_list = self.bm25.rank_batch(query_texts)
        
        # 步骤 2: C-BERT 批量重排
        print("Step 2: C-BERT is reranking candidates in batch...")
        reranked_results_list = self.cbert.rerank_batch(query_texts, bm25_candidate_ids_list)
        
        # 步骤 3: 批量分数融合
        print("Step 3: Fusing scores...")
        final_rankings = []
        for i in range(len(query_texts)):
            # 需要获取BM25的分数
            bm25_candidates = self.bm25.rank(query_texts[i], BM25_TOPN)
            bm25_scores_dict = {doc_id: score for doc_id, score in bm25_candidates}
            cbert_scores_dict = {doc_id: score for doc_id, score in reranked_results_list[i]}
            
            # 归一化
            bm25_norm = self._normalize_scores(bm25_scores_dict)
            cbert_norm = self._normalize_scores(cbert_scores_dict)
            
            ensemble_scores = {
                doc_id: ENSEMBLE_ALPHA * cbert_norm.get(doc_id, 0) + (1 - ENSEMBLE_ALPHA) * bm25_norm.get(doc_id, 0)
                for doc_id in bm25_scores_dict.keys()
            }
            
            final_ranking = sorted(ensemble_scores.items(), key=lambda item: item[1], reverse=True)
            final_rankings.append(final_ranking[:top_k])
        
        return final_rankings
        
    def _normalize_scores(self, scores_dict):
        if not scores_dict: return {}
        min_s, max_s = min(scores_dict.values()), max(scores_dict.values())
        if max_s == min_s: return {k: 1.0 for k in scores_dict}
        return {k: (v - min_s) / (max_s - min_s) for k, v in scores_dict.items()}

    # 保留单查询的rank方法用于非批量场景
    def rank(self, query_text, top_k):
        bm25_candidates = self.bm25.rank(query_text, BM25_TOPN)
        bm25_candidate_ids = [doc_id for doc_id, score in bm25_candidates]
        reranked_results = self.cbert.rerank_batch([query_text], [bm25_candidate_ids])[0]
        
        bm25_scores_dict = dict(bm25_candidates)
        cbert_scores_dict = dict(reranked_results)
        
        bm25_norm = self._normalize_scores(bm25_scores_dict)
        cbert_norm = self._normalize_scores(cbert_scores_dict)
        
        ensemble_scores = {
            doc_id: ENSEMBLE_ALPHA * cbert_norm.get(doc_id, 0) + (1 - ENSEMBLE_ALPHA) * bm25_norm.get(doc_id, 0)
            for doc_id in bm25_candidate_ids
        }
        
        final_ranking = sorted(ensemble_scores.items(), key=lambda item: item[1], reverse=True)
        return final_ranking[:top_k]