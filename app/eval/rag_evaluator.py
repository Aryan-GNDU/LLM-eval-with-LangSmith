from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.eval.rag_pipeline import rag_bot


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



correctness_instructions = """
You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER,
and the STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Grade the student answer based ONLY on its factual accuracy
relative to the ground truth answer.

(2) Ensure that the student answer does not contain any conflicting
statements.

(3) It is OK if the student answer contains more information than
the ground truth answer, as long as it is factually accurate
relative to the ground truth answer.

Correctness:

A correctness value of True means that the student's answer meets
all of the criteria.

A correctness value of False means that the student's answer does
not meet all of the criteria.

Explain your reasoning step-by-step to ensure your reasoning and
conclusion are correct.

Avoid simply stating the correct answer at the outset.
"""
relevance_instructions = """
You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION.

(2) Ensure the STUDENT ANSWER helps to answer the QUESTION.

Relevance:

A relevance value of True means that the student's answer meets
all of the criteria.

A relevance value of False means that the student's answer does
not meet all of the criteria.

Explain your reasoning step-by-step to ensure your reasoning and
conclusion are correct.

Avoid simply stating the correct answer at the outset.
"""

grounded_instructions = """
You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS.

(2) Ensure the STUDENT ANSWER does not contain hallucinated
information outside the scope of the FACTS.

Grounded:

A grounded value of True means that the student's answer meets
all of the criteria.

A grounded value of False means that the student's answer does
not meet all of the criteria.

Explain your reasoning step-by-step to ensure your reasoning and
conclusion are correct.

Avoid simply stating the correct answer at the outset.
"""

retrieval_relevance_instructions = """
You are a teacher grading a quiz.

You will be given a QUESTION and a set of FACTS provided by the student.

Here is the grade criteria to follow:

(1) Your goal is to identify FACTS that are completely unrelated
to the QUESTION.

(2) If the facts contain ANY keywords or semantic meaning related
to the question, consider them relevant.

(3) It is OK if the facts have SOME information that is unrelated
to the question as long as (2) is met.

Relevance:

A relevance value of True means that the FACTS contain ANY keywords
or semantic meaning related to the QUESTION and are therefore relevant.

A relevance value of False means that the FACTS are completely
unrelated to the QUESTION.

Explain your reasoning step-by-step to ensure your reasoning and
conclusion are correct.

Avoid simply stating the correct answer at the outset.
"""

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