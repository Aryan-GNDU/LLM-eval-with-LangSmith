def precision_at_k(retrieved_documents, relevant_chunk_ids, k=3):
    """
    Calculate Precision@K.

    Precision@K = relevant retrieved documents / K
    """

    retrieved_chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc in retrieved_documents[:k]
    ]

    relevant_retrieved = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in retrieved_chunk_ids
    )

    return relevant_retrieved / k


def recall_at_k(retrieved_documents, relevant_chunk_ids, k=3):
    """
    Calculate Recall@K.

    Recall@K = relevant retrieved documents / total relevant documents
    """

    retrieved_chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc in retrieved_documents[:k]
    ]

    relevant_retrieved = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in retrieved_chunk_ids
    )

    if not relevant_chunk_ids:
        return 0.0

    return relevant_retrieved / len(relevant_chunk_ids)


def hit_rate_at_k(retrieved_documents, relevant_chunk_ids, k=3):
    """
    Calculate Hit Rate@K.

    Returns 1 if at least one relevant document
    appears in the top K retrieved documents.
    Otherwise returns 0.
    """

    retrieved_chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc in retrieved_documents[:k]
    ]

    return int(
        any(
            chunk_id in relevant_chunk_ids
            for chunk_id in retrieved_chunk_ids
        )
    )