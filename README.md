# LangChain RAG Fundamentals

A beginner-friendly, step-by-step series of **Jupyter notebooks** covering the building blocks of a Retrieval-Augmented Generation (RAG) pipeline with LangChain. Each notebook explains one concept in plain language, runs a small example, and ends with key points to remember.

## Notebooks
| # | Notebook | Topic |
|---|---|---|
| 1 | `01_Data_Ingestion.ipynb` | Loading documents with LangChain document loaders (`TextLoader` and others) |
| 2 | `02_Text_Splitter.ipynb` | Splitting text into chunks: `chunk_size` and `chunk_overlap` |
| 3 | `03_Embeddings.ipynb` | Turning text into vectors with a free HuggingFace model or OpenAI, and comparing meanings with cosine similarity |
| 4 | `04_Vector_Store.ipynb` | Storing and searching embeddings in FAISS, then saving and loading the store |
| 5 | `05_Retriever.ipynb` | Turning a vector store into a retriever |
| 6 | `06_Chains.ipynb` | Prompt ➜ Model ➜ Output chains with LCEL |
| 7 | `07_LangSmith.ipynb` | Tracing and debugging runs with LangSmith (`@traceable`, `wrap_openai`) |

Supporting files: `sample.txt` (sample knowledge base) and `my_store/` (a saved FAISS index from notebook 4).

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your OpenAI / LangSmith keys
jupyter notebook
```
Notebooks 1–4 run fully offline with the free `all-MiniLM-L6-v2` embedding model. Notebooks 6–7 need an OpenAI key, and notebook 7 also needs a free LangSmith key.

## Tech stack
Python · LangChain · FAISS · HuggingFace sentence-transformers · OpenAI · LangSmith · Jupyter
