from app.rag.rag_pipeline import rag_bot
from app.eval.datasets.rag_datapoints import RETRIEVAL_GROUND_TRUTH
from app.eval.evaluators.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k
)

def evaluate_retrieval():
    for question, relevant_chunk_ids in RETRIEVAL_GROUND_TRUTH.items():

        result = rag_bot(question)

        score = precision_at_k(
            retrieved_documents=result["documents"],
            relevant_chunk_ids=relevant_chunk_ids,
            k=3,
        )

        recall_score = recall_at_k(
            retrieved_documents=result["documents"],
            relevant_chunk_ids=relevant_chunk_ids,
            k=3,
        )

        hit_rate_score = hit_rate_at_k(
            retrieved_documents=result["documents"],
            relevant_chunk_ids=relevant_chunk_ids,
            k=3,
        )

        print("\n" + "=" * 80)
        print("QUESTION:")
        print(question)

        print("\nEXPECTED RELEVANT CHUNKS:")
        print(relevant_chunk_ids)

        print("\nRETRIEVED CHUNKS:")
        for document in result["documents"][:3]:
            print(document.metadata.get("chunk_id"))

        print("\nPRECISION@3:")
        print(score)

        print("\nRECALL@3:")
        print(recall_score)

        print("\nHIT RATE@3:")
        print(hit_rate_score)


if __name__ == "__main__":
    evaluate_retrieval()