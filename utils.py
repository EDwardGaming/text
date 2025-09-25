# utils.py
import os
import json
from tqdm import tqdm
from config import TASK

def load_corpus(corpus_dir: str):
    """加载语料库文件夹中的所有json文件"""
    corpus = {}
    print(f"Loading corpus from: {corpus_dir}")
    for filename in tqdm(os.listdir(corpus_dir)):
        if filename.endswith('.json'):
            file_path = os.path.join(corpus_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # "id"：该法律文件的唯一标识符
                doc_id = data.get("id")
                corpus[doc_id] = data
    return corpus

def load_qrels(qrels_dir: str):
    """加载查询-相关文档对，适配每行有'document_id'和'relevant_documents'字段的jsonl"""
    qrels = {}
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(qrels_dir, f'{TASK}_{split}.jsonl')
        qrels[split] = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 适配格式：{"document_id": "...", "relevant_documents": [...]}
                query_id = item.get("document_id")
                doc_ids = item.get("relevant_documents", [])
                if query_id is not None:
                    qrels[split][query_id] = doc_ids
    return qrels