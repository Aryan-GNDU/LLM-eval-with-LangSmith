from openai import OpenAI
from app.constants.environ import GROQ_API_KEY



groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

eval_instructions = (
    "You are an expert professor specialized in "
    "grading students' answers to questions."
)


def correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
) -> bool:

    user_content = f"""You are grading the following question:

{inputs['question']}

Here is the real answer:

{reference_outputs['answer']}

You are grading the following predicted answer:

{outputs['response']}

Respond with CORRECT or INCORRECT:

Grade:
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": eval_instructions
            },
            {
                "role": "user",
                "content": user_content
            },
        ],
    ).choices[0].message.content

    print("Judge response:", repr(response))

    return "CORRECT" in response.strip().upper()

def concision(outputs: dict, reference_outputs: dict) -> bool:
    return len(outputs["response"]) < 2 * len(reference_outputs["answer"])


if __name__ == "__main__":

    inputs = {
        "question": "What is LangChain?"
    }

    outputs = {
        "response": "A framework for building LLM applications"
    }

    reference_outputs = {
        "answer": "A framework for building LLM applications"
    }

    correctness_result = correctness(
        inputs,
        outputs,
        reference_outputs
    )

    concision_result = concision(
        outputs,
        reference_outputs
    )

    print(f"Correctness: {correctness_result}")
    print(f"Concision: {concision_result}")