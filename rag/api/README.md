# RAG API Documentation

This FastAPI server wraps the RAG pipeline in `rag_demo.py` and exposes endpoints to index PDFs, search, build prompts, and request an LLM answer. Figure images are extracted, captioned, cached, and served over HTTP so a frontend can display them.

- Server code: `rag/api/server.py`
- Static files (cache): mounted at `/cache` and served from the local `.rag_cache/` directory
- Images policy: figures only, saved under `.rag_cache/<dataset_key>/images/`

## How to run

Option A: using uvicorn

```bash
python -m uvicorn rag.api.server:app --host 0.0.0.0 --port 8000
```

Option B: run directly

```bash
python rag/api/server.py
```

Open API docs in your browser:

- http://localhost:8000/docs

## Environment variables

- `NEO4J_URI` (default: `bolt://localhost:7687`)
- `NEO4J_USER` (default: `neo4j`)
- `NEO4J_PASSWORD` (default: `password`)
- `GROQ_API_KEY` (required for LLM)
- `CORS_ALLOW_ORIGINS` (default: `*`)

## Endpoints

### POST /index
Index one or more PDF files and build the cache. Figures are extracted to `.rag_cache/<dataset_key>/images/` and captioned via the Granite Docling VLM.

Request body JSON:

```json
{
  "pdf_files": ["1512.03385v1.pdf"],
  "use_section_chunker": true,
  "section_tag_keywords": [],
  "use_docling_chunker": true,
  "target_chars": 900,
  "overlap_chars": 150,
  "text_model": "Alibaba-NLP/gte-multilingual-base",
  "vlm_caption_model": "ibm-granite/granite-docling-258M",
  "use_neo4j": true,
  "neo4j_uri": "bolt://localhost:7687",
  "neo4j_user": "neo4j",
  "neo4j_password": "password"
}
```

Response JSON (excerpt):

```json
{
  "dataset_key": "<MD5>",
  "cached": false,
  "n_chunks": 123,
  "n_text": 87,
  "n_image": 36,
  "meta": { "images_saved": "figures_only", "chunker": "section", ... }
}
```

### POST /upload-index
Upload and index PDF files via multipart form-data (no need for local file paths in the request). Files are saved under `.rag_cache/uploads/<pid>/`, then indexed. Figures are saved under `.rag_cache/<dataset_key>/images/`.

Form fields:

- `files`: one or more PDF files (repeat field for multiple files)
- `use_section_chunker` (default true)
- `section_tag_keywords`: comma-separated string (e.g., `customer, loan`)
- `use_docling_chunker` (default true)
- `target_chars` (default 900)
- `overlap_chars` (default 150)
- `text_model` (default Alibaba-NLP/gte-multilingual-base)
- `vlm_caption_model` (default ibm-granite/granite-docling-258M)
- `use_neo4j` (default true)
- `neo4j_uri`, `neo4j_user`, `neo4j_password` (optional)

### GET /search
Search previously indexed data.

Query params:

- `dataset_key` (required)
- `q` (required)
- `top_k` (default 5)
- `alpha` (default 0.6)
- `use_neo4j_vector` (default true)
- `neo4j_uri`, `neo4j_user`, `neo4j_password` (optional overrides)

Response hits include `image_url` (HTTP path under `/cache/...`) and `image_path` (local path) when available.

### GET /prompt
Build the prompt with context snippets and image links.

Query params: `dataset_key`, `q`, `top_k`, `alpha`

### POST /ask
Trigger the LLM answer (Groq) using the built prompt. The answer streams to server logs.

Query params: `dataset_key`, `q`, `top_k`, `alpha`

### Static: /cache
Static mount for `.rag_cache/`. Images are accessible as:

```
/cache/<dataset_key>/images/<file>.png
```

## curl examples

Replace `<DATASET_KEY>` with the value returned by `/index`.

### Index PDFs

PowerShell:

```powershell
curl -Method POST http://localhost:8000/index -ContentType "application/json" -Body '{
  "pdf_files": ["1512.03385v1.pdf"],
  "use_section_chunker": true,
  "use_docling_chunker": true
}'
```

bash:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_files": ["1512.03385v1.pdf"],
    "use_section_chunker": true,
    "use_docling_chunker": true
  }'
```

### Upload and index PDFs (multipart)

Single file:

```bash
curl -X POST http://localhost:8000/upload-index \
  -H "Accept: application/json" \
  -F "files=@1512.03385v1.pdf;type=application/pdf" \
  -F "use_section_chunker=true" \
  -F "use_docling_chunker=true"
```

Multiple files:

```bash
curl -X POST http://localhost:8000/upload-index \
  -H "Accept: application/json" \
  -F "files=@1512.03385v1.pdf;type=application/pdf" \
  -F "files=@2303.08774v6.pdf;type=application/pdf" \
  -F "section_tag_keywords=customer,loan" \
  -F "target_chars=900" -F "overlap_chars=150"
```

PowerShell (Invoke-RestMethod):

```powershell
$form = @{
  files = Get-Item "1512.03385v1.pdf"
  use_section_chunker = 'true'
  use_docling_chunker = 'true'
}
Invoke-RestMethod -Uri http://localhost:8000/upload-index -Method Post -Form $form
```

### Search

```bash
curl "http://localhost:8000/search?dataset_key=<DATASET_KEY>&q=weight%20layer%20%3F&top_k=5&alpha=0.6"
```

### Prompt

```bash
curl "http://localhost:8000/prompt?dataset_key=<DATASET_KEY>&q=weight%20layer%20%3F&top_k=5&alpha=0.6"
```

### Ask LLM (streams to server logs)

```bash
curl -X POST "http://localhost:8000/ask?dataset_key=<DATASET_KEY>&q=weight%20layer%20%3F&top_k=5&alpha=0.6"
```

### Fetch an image

If a hit returns `"image_url": "/cache/<dataset_key>/images/1512.03385v1_figure_0.png"`:

```bash
curl -o figure0.png "http://localhost:8000/cache/<dataset_key>/images/1512.03385v1_figure_0.png"
```
