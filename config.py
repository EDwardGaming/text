# config.py (优化后)
import os

# --- 性能相关参数 ---
BM25_TOPN = 500                  # BM25 初筛候选数（再由BERT重排到TOP_K）
NUM_WORKERS_BM25 = 14             # （可选）BM25 多进程并行的进程数

# --- 路径配置 ---
BASE_DIR = ".."
TASK = 'eu2uk'

if TASK == 'eu2uk':
    CORPUS_DIR = f"{BASE_DIR}/datasets/eu_corpus"
    DOC_CORPUS_DIR = f"{BASE_DIR}/datasets/uk_corpus"
    QRELS_DIR = f"{BASE_DIR}/datasets/eu2uk"
else: # uk2eu
    CORPUS_DIR = f"{BASE_DIR}/datasets/uk_corpus"
    DOC_CORPUS_DIR = f"{BASE_DIR}/datasets/eu_corpus"
    QRELS_DIR = f"{BASE_DIR}/datasets/uk2eu"

# --- 模型配置 ---
CBERT_MODEL_NAME = './finetune_cbert/c-bert-finetuned/best_model' 
LEGAL_BERT_MODEL_NAME = './legal_bert/legal-bert-base-uncased' 
DEVICE = 'cuda'

# --- 检索配置 ---
TOP_K = 100
BM25_K1 = 8.0 
BM25_B = 1.0
YEAR_FILTER_RANGE = {"lower": 5, "upper": 10} 
ENSEMBLE_ALPHA = 0.6 

# --- 嵌入和索引文件路径 (关键修改) ---
# 让索引文件名与模型路径相关联，避免混用
def get_index_paths(model_name_or_path):
    # 从路径中提取一个安全的文件名部分
    model_name_safe = os.path.basename(model_name_or_path.strip('/\\'))
    base_path = f'../datasets/{TASK}/{model_name_safe}'
    return {
        "embeddings": f'{base_path}_embeddings.npy',
        "faiss_index": f'{base_path}_faiss.index',
        "doc_ids": f'{base_path}_doc_ids.json'
    }