from dotenv import load_dotenv
from langsmith import Client

from app.eval.rag_pipeline import rag_bot
from app.eval.rag_evaluator import (
    correctness,
    relevance,
    groundedness,
    retrieval_relevance,
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
            correctness,
            groundedness,
            relevance,
            retrieval_relevance,
        ],
        experiment_prefix="groq-gpt-oss-20b-rag",
        metadata={
            "model": "openai/gpt-oss-20b",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        max_concurrency=1,
    )

    print("\nEvaluation completed!")
    print(experiment_results)