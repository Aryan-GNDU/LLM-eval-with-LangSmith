from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.rag.rag_pipeline import rag_bot
from app.eval.evaluators.instructions import (
    correctness_instructions,
    relevance_instructions,
    grounded_instructions,
    retrieval_relevance_instructions,
)

load_dotenv(dotenv_path=".env", override=True)


class CorrectnessGrade(TypedDict):
    explanation: Annotated[
        str,
        ...,
        "Explain your reasoning for the score",
    ]

    correct: Annotated[
        bool,
        ...,
        "True if the answer is correct, False otherwise.",
    ]

class RelevanceGrade(TypedDict):
    explanation: Annotated[
        str,
        ...,
        "Explain your reasoning for the score",
    ]

    relevant: Annotated[
        bool,
        ...,
        "Provide the score on whether the answer addresses the question",
    ]

class GroundedGrade(TypedDict):
    explanation: Annotated[
        str,
        ...,
        "Explain your reasoning for the score",
    ]

    grounded: Annotated[
        bool,
        ...,
        "Provide the score on if the answer hallucinates from the documents",
    ]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[
        str,
        ...,
        "Explain your reasoning for the score",
    ]

    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, "
        "False otherwise",
    ]


grader_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
).with_structured_output(
    CorrectnessGrade,
    method="json_schema",
)


relevance_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
).with_structured_output(
    RelevanceGrade,
    method="json_schema",
)

grounded_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
).with_structured_output(
    GroundedGrade,
    method="json_schema",
)

retrieval_relevance_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
).with_structured_output(
    RetrievalRelevanceGrade,
    method="json_schema",
)


def correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> bool:
    """Evaluate the factual correctness of a RAG answer."""

    answers = f"""
QUESTION: {inputs['question']}

GROUND TRUTH ANSWER: {reference_outputs['answer']}

STUDENT ANSWER: {outputs['answer']}
"""

    grade = grader_llm.invoke(
        [
            {
                "role": "system",
                "content": correctness_instructions,
            },
            {
                "role": "user",
                "content": answers,
            },
        ]
    )

    print("Judge explanation:", grade["explanation"])
    print("Judge score:", grade["correct"])

    return grade["correct"]

def relevance(
    inputs: dict,
    outputs: dict,
) -> bool:
    """Evaluate whether the RAG answer addresses the question."""

    answer = (
        f"QUESTION: {inputs['question']}\n"
        f"STUDENT ANSWER: {outputs['answer']}"
    )

    grade = relevance_llm.invoke(
        [
            {
                "role": "system",
                "content": relevance_instructions,
            },
            {
                "role": "user",
                "content": answer,
            },
        ]
    )

    print("Relevance explanation:", grade["explanation"])
    print("Relevance score:", grade["relevant"])

    return grade["relevant"]

def groundedness(
    inputs: dict,
    outputs: dict,
) -> bool:
    """Evaluate whether the answer is supported by retrieved documents."""

    doc_string = "\n\n".join(
        doc.page_content
        for doc in outputs["documents"]
    )

    answer = (
        f"FACTS:\n{doc_string}\n\n"
        f"STUDENT ANSWER:\n{outputs['answer']}"
    )

    grade = grounded_llm.invoke(
        [
            {
                "role": "system",
                "content": grounded_instructions,
            },
            {
                "role": "user",
                "content": answer,
            },
        ]
    )

    print("Groundedness explanation:", grade["explanation"])
    print("Groundedness score:", grade["grounded"])

    return grade["grounded"]

def retrieval_relevance(
    inputs: dict,
    outputs: dict,
) -> bool:
    """Evaluate whether retrieved documents are relevant to the question."""

    doc_string = "\n\n".join(
        doc.page_content
        for doc in outputs["documents"]
    )

    answer = (
        f"FACTS:\n{doc_string}\n\n"
        f"QUESTION:\n{inputs['question']}"
    )

    grade = retrieval_relevance_llm.invoke(
        [
            {
                "role": "system",
                "content": retrieval_relevance_instructions,
            },
            {
                "role": "user",
                "content": answer,
            },
        ]
    )

    print(
        "Retrieval relevance explanation:",
        grade["explanation"],
    )

    print(
        "Retrieval relevance score:",
        grade["relevant"],
    )

    return grade["relevant"]



if __name__ == "__main__":
    question = "What is prompt engineering?"

    print("\n--- RUNNING RAG BOT ---")

    rag_result = rag_bot(question)

    print("\nRAG ANSWER:")
    print(rag_result["answer"])

    test_inputs = {
        "question": question
    }

    print("\n--- RELEVANCE TEST ---")

    relevance_result = relevance(
        test_inputs,
        rag_result,
    )

    print("Final relevance:", relevance_result)

    print("\n--- GROUNDEDNESS TEST ---")

    groundedness_result = groundedness(
        test_inputs,
        rag_result,
    )

    print("Final groundedness:", groundedness_result)

    print("\n--- RETRIEVAL RELEVANCE TEST ---")

    retrieval_result = retrieval_relevance(
        test_inputs,
        rag_result,
    )

    print(
        "Final retrieval relevance:",
        retrieval_result,
    )