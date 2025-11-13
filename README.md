# RAG

## Structure
- `rag/types.py`
- `rag/loaders/doc_loader.py`
- `rag/chunking/semantic_chunker.py`
- `rag/embeddings/text_embedder.py`
- `rag/embeddings/image_embedder.py`
- `rag/retrieval/hybrid_indexer.py`
- `rag/llm/groq_client.py`
- `rag/prompts/citation_prompt.py`
- `rag/graph/kg.py`
- `rag/graph/graph_flow.py`
- `scripts/run_rag.py`

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
```

Set env var:
```
set GROQ_API_KEY=your_key_here
```

## Run
```
python scripts/run_rag.py
```

## Notes
- Docling is used to parse PDFs and extract images. Image embeddings use a CLIP model (`clip-ViT-B-32`). If you prefer a different image embedding, we can switch.
- Text embeddings use `Alibaba-NLP/gte-multilingual-base`.
- Simple KG flow with LangGraph demonstrates extracting trivial triples and updating a `networkx` graph.
