# evaluate.py
def recall_at_k(results, qrels, k):
    """计算 Recall@k"""
    total_queries = len(qrels)
    if total_queries == 0:
        return 0.0
    
    recall_sum = 0
    for query_id, ranked_list in results.items():
        if query_id in qrels:
            retrieved_ids = {doc_id for doc_id, score in ranked_list[:k]}
            relevant_ids = set(qrels[query_id])
            
            hits = len(retrieved_ids.intersection(relevant_ids))
            recall_sum += hits / len(relevant_ids) if relevant_ids else 1.0

    return recall_sum / total_queries