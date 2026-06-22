import streamlit as st

from src.rag_pipeline import (
    answer_question
)

st.title(
    "CrediTrust Complaint Assistant"
)

question = st.text_input(
    "Ask a question"
)

if st.button("Ask"):

    answer, sources = (
        answer_question(question)
    )

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Sources")

    for i, source in enumerate(sources):

        st.write(
            f"Source {i+1}"
        )

        st.info(source)

if st.button("Clear"):

    st.rerun()