from openai import OpenAI
from langsmith import Client
from app.constants.environ import GROQ_API_KEY

from app.eval.evaluators.llm_as_judge import correctness, concision


openai_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

client = Client()


default_instructions = (
    "Respond to the user's question in a short, "
    "concise manner."
)


def my_app(
    question: str,
    model: str = "openai/gpt-oss-20b",
    instructions: str = default_instructions,
) -> str:

    response = openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response.choices[0].message.content


def ls_target(inputs: dict) -> dict:
    return {
        "response": my_app(inputs["question"])
    }


if __name__ == "__main__":

    dataset_name = "Chatbot Evaluation"

    experiment_results = client.evaluate(
        ls_target,
        data=dataset_name,
        evaluators=[
            correctness,
            concision,
        ],
        experiment_prefix="groq-gpt-oss-20b-chatbot",
    )

    print(experiment_results)