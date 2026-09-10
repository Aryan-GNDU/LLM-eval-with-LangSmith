from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()


def create_dataset_and_examples():
    dataset_name = "Chatbot Evaluation"

    dataset = client.create_dataset(dataset_name)

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[
            {"question": "What is LangChain?"},
            {"question": "What is LangSmith?"},
            {"question": "What is OpenAI?"},
            {"question": "What is Google?"},
            {"question": "What is Mistral?"},
            {"question": "What is Cohere?"},
        ],
        outputs=[
            {"answer": "A framework for building LLM applications"},
            {"answer": "A platform for observing and evaluating LLM applications"},
            {"answer": "A company that creates Large Language Models"},
            {"answer": "A technology company known for search"},
            {"answer": "A company that creates Large Language Models"},
            {"answer": "A company that creates Large Language Models"},
        ],
    )


if __name__ == "__main__":
    create_dataset_and_examples()