from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

def load_documents():
    docs = [WebBaseLoader(url).load() for url in urls]

    docs_list = [
        item
        for sublist in docs
        for item in sublist
    ]

    return docs_list


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0,
    )

    doc_splits = text_splitter.split_documents(docs)

    for index, doc in enumerate(doc_splits):
        doc.metadata["chunk_id"] = f"chunk_{index:04d}"

    return doc_splits


def create_vectorstore(doc_splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )

    return vectorstore


def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

def create_reranked_retriever(retriever):
    reranker = FlashrankRerank(
        top_n=3
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever,
    )


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


@traceable()
def rag_bot(question: str) -> dict:
    docs = retriever.invoke(question)

    docs_string = " ".join(
        doc.page_content
        for doc in docs
    )

    instructions = f"""
You are a helpful assistant who is good at
analyzing source information and answering questions.

Use the following source documents to answer the user's question.

If you don't know the answer, just say that you don't know.

Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}
"""

    ai_msg = llm.invoke(
        [
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": question,
            },
        ]
    )

    return {
        "answer": ai_msg.content,
        "documents": docs,
    }


docs = load_documents()

doc_splits = split_documents(docs)

vectorstore = create_vectorstore(doc_splits)

base_retriever = create_retriever(vectorstore)
retriever = create_reranked_retriever(base_retriever)


# if __name__ == "__main__":
#     result = rag_bot(
#         "What are five types of adversarial attacks?"
#     )

#     print("\nANSWER:")
#     print(result["answer"])

#     print("\nRETRIEVED DOCUMENTS:")

#     for i, document in enumerate(result["documents"], start=1):
#         print(f"\n--- Document {i} ---")
#         print("Chunk ID:", document.metadata.get("chunk_id"))
#         print("Source:", document.metadata.get("source"))
#         print(document.page_content[:500])


if __name__ == "__main__":
    keywords = [
        "Majority label bias",
        "Recency bias",
        "Common token bias",
        "Token Manipulation",
        "Gradient based Attacks",
        "Jailbreak Prompting",
        "Human red-teaming",
        "Model red-teaming",
    ]

    for keyword in keywords:
        print(f"\n{'=' * 80}")
        print(f"SEARCHING: {keyword}")
        print(f"{'=' * 80}")

        matches = [
            doc
            for doc in doc_splits
            if keyword.lower() in doc.page_content.lower()
        ]

        for doc in matches:
            print("\nChunk ID:", doc.metadata.get("chunk_id"))
            print("Source:", doc.metadata.get("source"))
            print(doc.page_content)