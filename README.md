# PaperMind — Multilingual Document Intelligence System

PaperMind is a RAG (Retrieval Augmented Generation) system that lets you upload
documents in any language and ask questions about them in any language.
It retrieves the most relevant passages from your documents, generates grounded
answers using Claude, and shows exactly which document and page each answer
came from.

> 🚧 This project is currently under active development.

---

## What it does

- Upload any PDF or image/screenshot of a document
- Ask questions in any language — system responds in the same language
- Answers are grounded in your documents, not Claude's general knowledge
- Shows source attribution: document name and page number with every answer
- Side-by-side PDF viewer — select text with your mouse to ask about it
- Remembers conversation context across multi-turn questions
- Works across multiple documents simultaneously in one session

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Claude Sonnet (answers) + Claude Haiku (vision/routing) |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 (local, free, 50+ languages) |
| Vector Store | ChromaDB (persistent local storage) |
| Document Loading | PyMuPDF |
| Framework | LangChain (text splitting + memory) |
| Backend | FastAPI |
| Frontend | HTML + PDF.js |
<!-- | Evaluation | RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall) | -->
<!-- | Deployment | Docker | -->

---
## Setup

**1. Clone and create environment**
```bash
git clone https://github.com/dixitdevarshi/PaperMind.git
cd PaperMind
conda create -n papermind python=3.10 -y
conda activate papermind
pip install -r requirements.txt
```

**2. Add your API key**
```bash
# Create a .env file
echo ANTHROPIC_API_KEY=your_key_here > .env
```

**3. Run**
```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser.

---

<!-- ## Docker

```bash
docker build -t papermind .
docker run -p 8000:8000 --env-file .env papermind
``` -->