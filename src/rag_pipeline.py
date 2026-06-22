from transformers import pipeline

from retriever import (
    ComplaintRetriever
)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

retriever = ComplaintRetriever()


def answer_question(question):

    results = retriever.retrieve(question)

    docs = results["documents"][0]

    context = "\n\n".join(docs)

    prompt = f"""
    You are a financial analyst assistant.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = generator(
        prompt,
        max_new_tokens=200
    )

    return (
        response[0]["generated_text"],
        docs
    )