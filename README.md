# Intelligent Complaint Analysis for Financial Services: Building a RAG-Powered Chatbot

## Overview

This project develops a Retrieval-Augmented Generation (RAG) chatbot for CrediTrust Financial. The chatbot helps stakeholders analyze customer complaints across four financial products:

* Credit Cards
* Personal Loans
* Savings Accounts
* Money Transfers

Instead of manually reviewing thousands of complaint narratives, users can ask questions in natural language and receive evidence-based answers generated from relevant complaint data.

---

## Business Objective

CrediTrust receives a large volume of customer complaints, making it difficult for Product Managers, Customer Support teams, and Compliance teams to identify trends quickly.

The goal of this project is to:

* Reduce the time required to analyze complaints
* Help non-technical stakeholders access insights
* Identify recurring customer issues
* Support data-driven decision making

---

## Project Tasks

### Task 1: Data Preprocessing and EDA

* Analyze complaint distribution across products
* Analyze complaint narrative lengths
* Identify missing narratives
* Filter data to target products
* Clean and normalize complaint text

Output:

```text
data/processed/filtered_complaints.csv
```

### Task 2: Embedding Pipeline

* Create a stratified sample
* Chunk complaint narratives
* Generate embeddings using Sentence Transformers
* Store embeddings in ChromaDB

Output:

```text
vector_store/chroma_db/
```

### Task 3: RAG Pipeline

* Retrieve relevant complaint chunks
* Build prompt templates
* Generate answers using FLAN-T5
* Evaluate retrieval and generation quality

### Task 4: Interactive Chat Interface

* Streamlit-based chatbot
* User question input
* AI-generated answers
* Source document display
* Clear chat functionality

---

## Project Structure

```text
rag-complaint-chatbot/

├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_chunking_embeddings.ipynb
│   └── 03_rag_demo.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── sampling.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── generator.py
│   └── rag_pipeline.py
│
├── vector_store/
│
├── tests/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Sentence Transformers
* LangChain
* ChromaDB
* Transformers (FLAN-T5)
* Streamlit

---

## Installation

```bash
git clone <repository-url>
cd rag-complaint-chatbot

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the Project

### Preprocess Data

```bash
python src/preprocessing.py
```

### Create Sample Dataset

```bash
python src/sampling.py
```

### Build Vector Store

```bash
python src/vector_store.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## Future Improvements

* Better embedding models
* Hybrid search (keyword + semantic search)
* Conversation memory
* Advanced evaluation metrics
* Larger language models

---
