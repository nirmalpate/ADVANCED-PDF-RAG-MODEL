# ADVANCED-PDF-RAG-MODEL

# Advanced PDF Question Answering with RAG

Ask questions about your own PDFs. This project uses **Retrieval-Augmented Generation (RAG)** with **hybrid search** and a **reranker**, so answers come from your documents, with the file and page of each source.

Runs fully in **Google Colab** with open-source models. No paid API needed.

## How It Works

```
PDFs -> Load pages -> Split into chunks -> Embed -> Chroma (vectors) + BM25 (keywords)
                                                          |
Question -> Hybrid retrieval (top 5 + 5) -> Reranker (best 3) -> LLM + prompt -> Answer + sources
```

1. **Load**: `PyPDFLoader` reads each PDF page by page.
2. **Chunk**: `RecursiveCharacterTextSplitter` splits text into 500-character chunks with 50 overlap.
3. **Index**: chunks are embedded (`all-MiniLM-L6-v2`) into Chroma and also indexed by BM25.
4. **Hybrid retrieval**: keyword search (BM25, 40%) and semantic search (vectors, 60%) are combined.
5. **Rerank**: a cross-encoder (`bge-reranker-base`) keeps the 3 most relevant chunks.
6. **Generate**: a local LLM answers using only those chunks.
7. **Cite**: the file name and page number of each source are printed.

## Tech Stack

| Part | Tool |
|---|---|
| Framework | LangChain |
| PDF loading | pypdf |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Keyword search | BM25 (`rank_bm25`) |
| Reranker | `BAAI/bge-reranker-base` |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` (Hugging Face Transformers) |
| Environment | Google Colab (T4 GPU) |

## Setup

1. Open Google Colab and upload `advanced_pdf_rag.ipynb` (File -> Upload notebook).
2. Switch to a GPU: **Runtime -> Change runtime type -> T4 GPU**.
3. Upload your PDFs into the `pdf_data` folder (Colab sidebar -> Files).
4. Run all cells in order. The first run downloads three models, so it takes a few minutes.

## Usage

```python
ask("What is this document about?")
```

Output:

```
Answer: ...

Sources:
- report.pdf | page 3
- report.pdf | page 7
```

## Notebook Structure

| Step | Purpose |
|---|---|
| 1 | Install libraries |
| 2 | Imports |
| 3 | Create `pdf_data` folder |
| 4 | Load and chunk PDFs |
| 5 | Embeddings and Chroma vector store |
| 6 | Hybrid retriever + reranker |
| 7 | Load the LLM |
| 8 | Build the RAG chain |
| 9 | Ask questions with `ask()` |

## Configuration

| Setting | Where | Default |
|---|---|---|
| Chunk size / overlap | `load_and_chunk(...)` | 500 / 50 |
| Hybrid weights | `EnsembleRetriever(weights=...)` | BM25 0.4, vectors 0.6 |
| Candidates per retriever | `bm25.k`, `search_kwargs` | 5 |
| Final chunks sent to LLM | `CrossEncoderReranker(top_n=...)` | 3 |
| Max answer length | `max_new_tokens` | 256 |
| LLM | `MODEL_NAME` | Qwen2.5-1.5B-Instruct |

For stronger answers, try `Qwen/Qwen2.5-3B-Instruct` (GPU needed).

## Troubleshooting

- **`No module named 'langchain.chains'`**: use `langchain_classic` imports (already done in the notebook).
- **Cell runs forever**: you are on CPU. Switch to a T4 GPU.
- **`NameError: rag_chain`**: run the chain-building cell before `ask()`.
- **`temperature` warning**: harmless, already handled with `temperature=None`.
- **No PDFs found**: upload files into `pdf_data` and re-run from step 4.
- **Pip dependency warnings about `google-colab`**: safe to ignore.

## Limitations

- Small LLMs can still give incomplete answers.
- Scanned PDFs (images) need OCR, which is not included.
- The vector database lives in memory and is rebuilt each session.

## Future Improvements

- Conversation memory and query rewriting
- Evaluation with RAGAS (faithfulness, answer relevancy)
- Agentic RAG with LangGraph (retry, web search fallback)
- Gradio or Streamlit chat UI
- OCR for scanned PDFs
