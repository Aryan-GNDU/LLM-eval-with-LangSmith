from app.eval.datasets.rag_datapoints import RETRIEVAL_GROUND_TRUTH



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


def reciprocal_rank_at_k(retrieved_documents, relevant_chunk_ids, k=3):
    """
    Calculate Reciprocal Rank@K.

    Returns the reciprocal of the rank of the first
    relevant document in the top K results.

    Returns 0 if no relevant document is found.
    """

    retrieved_chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc in retrieved_documents[:k]
    ]

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1 / rank

    return 0.0

def precision_evaluator(inputs, outputs):
    question = inputs["question"]

    relevant_chunk_ids = RETRIEVAL_GROUND_TRUTH[question]

    score = precision_at_k(
        retrieved_documents=outputs["documents"],
        relevant_chunk_ids=relevant_chunk_ids,
        k=3,
    )

    return {
        "key": "precision_at_3",
        "score": score,
    }



def recall_evaluator(inputs, outputs):
    question = inputs["question"]

    relevant_chunk_ids = RETRIEVAL_GROUND_TRUTH[question]

    score = recall_at_k(
        retrieved_documents=outputs["documents"],
        relevant_chunk_ids=relevant_chunk_ids,
        k=3,
    )

    return {
        "key": "recall_at_3",
        "score": score,
    }


def hit_rate_evaluator(inputs, outputs):
    question = inputs["question"]

    relevant_chunk_ids = RETRIEVAL_GROUND_TRUTH[question]

    score = hit_rate_at_k(
        retrieved_documents=outputs["documents"],
        relevant_chunk_ids=relevant_chunk_ids,
        k=3,
    )

    return {
        "key": "hit_rate_at_3",
        "score": score,
    }


def mrr_evaluator(inputs, outputs):
    question = inputs["question"]

    relevant_chunk_ids = RETRIEVAL_GROUND_TRUTH[question]

    score = reciprocal_rank_at_k(
        retrieved_documents=outputs["documents"],
        relevant_chunk_ids=relevant_chunk_ids,
        k=3,
    )

    return {
        "key": "mrr_at_3",
        "score": score,
    }