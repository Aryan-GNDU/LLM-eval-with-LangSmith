from dotenv import load_dotenv
from langsmith import Client

load_dotenv(dotenv_path=".env", override=True)

client = Client()

examples = [
    {
        "inputs": {
            "question": "How does the ReAct agent use self-reflection?"
        },
        "outputs": {
            "answer": "ReAct integrates reasoning and acting, performing actions - such tools like Wikipedia search API - and then observing / reasoning about the tool outputs."
        },
    },
    {
        "inputs": {
            "question": "What are the types of biases that can arise with few-shot prompting?"
        },
        "outputs": {
            "answer": "The biases that can arise with few-shot prompting include (1) Majority label bias, (2) Recency bias, and (3) Common token bias."
        },
    },
    {
        "inputs": {
            "question": "What are five types of adversarial attacks?"
        },
        "outputs": {
            "answer": "Five types of adversarial attacks are (1) Token manipulation, (2) Gradient based attack, (3) Jailbreak prompting, (4) Human red-teaming, (5) Model red-teaming."
        },
    },
]


RETRIEVAL_GROUND_TRUTH = {
    "How does the ReAct agent use self-reflection?": [
        "chunk_0005",
        "chunk_0007",
    ],

    "What are the types of biases that can arise with few-shot prompting?": [
        "chunk_0066",
    ],

    "What are five types of adversarial attacks?": [
        "chunk_0113",
    ],
}


dataset_name = "RAG Test Evaluation"


if __name__ == "__main__":
    dataset = client.create_dataset(dataset_name=dataset_name)

    client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
    )
    