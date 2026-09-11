from dotenv import load_dotenv
from langsmith import Client

from app.rag.rag_pipeline import rag_bot

from app.eval.evaluators.retrieval_metrics import (
    precision_evaluator,
    recall_evaluator,
    hit_rate_evaluator,
    mrr_evaluator,
)


load_dotenv(dotenv_path=".env", override=True)

client = Client()

dataset_name = "RAG Test Evaluation"


def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])


if __name__ == "__main__":

    experiment_results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[
            precision_evaluator,
            recall_evaluator,
            hit_rate_evaluator,
            mrr_evaluator,
        ],
        experiment_prefix="groq-gpt-oss-20b-retrieval",
        metadata={
            "model": "openai/gpt-oss-20b",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "retriever": "InMemoryVectorStore",
            "reranker": "FlashRank",
            "retrieval_k": 6,
            "rerank_top_n": 3,
        },
        max_concurrency=1,
    )

    print("\nRetrieval evaluation completed!")
    print(experiment_results)
