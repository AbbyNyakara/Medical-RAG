# Medical RAG Service

A Retrieval-Augmented Generation (RAG) system for querying PDF medical documents using FastAPI, LangChain, and OpenAI.

## Features

- 📄 **PDF Ingestion**: Extract and process medical documents
- 🔍 **Semantic Search**: Retrieve relevant context from documents using embeddings
- 🤖 **LLM Integration**: Generate answers using GPT-4o-mini
- 🚀 **REST API**: Simple FastAPI endpoint for querying
- 💾 **Vector Database**: ChromaDB for efficient document storage and retrieval

## The Data flow: 

Extractor Text (OCR)-> Chunking -> Embedding -> Storage (Pinecone) -> User Asks Question -> Embed User Query (Vectorize) -> Similarity Search of vector store to find close vectors+Associated content -> Pass retrieved content onto Prompt template (Hydrating)-> Pass hydrated prompt to LLM Return Response to user

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- OpenAI API key

## Installation

### 1. Clone and Navigate to Project

```bash
git clone https://github.com/AbbyNyakara/LangChain-Projects.git
cd medical-rag-service

```

### 2. Install Dependencies

```bash
uv sync
```

This installs all dependencies from `pyproject.toml`, including:

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `langchain` - LLM orchestration
- `openai` - OpenAI API client
- `Pinecone` - Vector database
- `amazon textract` - PDF extraction / OCR
- `python-dotenv` - Environment variables

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

**To get your API key:**

1. Go to [OpenAI API Keys](https://platform.openai.com/api/keys)
2. Create a new secret key
3. Copy and paste it into `.env`

### 4. Prepare Sample Data (Optional)

Place a PDF in the `data/` directory:

```bash
mkdir -p data
# Copy your PDF to data/ directory
cp /path/to/your/document.pdf data/
```

Or use the included sample: `data/fake-data.pdf`

## Running the Application

### Start the API Server

```bash
uv run python api/main.py
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Access the API

**Interactive Documentation (Swagger UI):**

- Open http://localhost:8000/docs in your browser
- Test endpoints directly in the UI

**Or use curl:**

#### Query the RAG

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the patient symptoms?"}'
```

**Response:**

```json
{
  "answer": "The patient presents with...",
  "num_sources": 3
}
```

### Stop the Server

Press `Ctrl + C` in the terminal

## First-Time Setup Workflow

### Step 1: Install & Configure

```bash
# Install dependencies
uv sync

# Create .env file with OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Step 2: Start the API

```bash
uv run python api/main.py
```

## API Endpoints

### `POST /query`

Query the RAG system with a question.

**Request:**

```json
{
  "question": "What are the patient's symptoms?"
}
```

**Response:**

```json
{
  "answer": "The patient presents with...",
  "num_sources": 3
}
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'generation'`

**Solution:** Make sure you're running from the project root:

```bash
cd /Users/abigaelmogusu/projects/LangChain-Projects/medical-rag-service
uv run python api/main.py
```

### `openai.error.AuthenticationError`

**Solution:** Check your `.env` file has a valid API key:

```bash
cat .env  # Should show OPENAI_API_KEY=sk-...
```

### Vector store is empty

**Solution:** Ingest a PDF first using the `/ingest` endpoint or:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "./data/fake-aps.pdf"}'
```

### Port 8000 already in use

**Solution:** Use a different port:

```bash
uv run python api/main.py --port 8001
```

## Development

### Run with Auto-Reload

The API already has `reload=True` enabled. Code changes will automatically restart the server.

### Add More Endpoints

Edit `api/main.py` to add new endpoints:

### Modify Prompts

Edit `src/generation/prompt_template.py` to customize the medical assistant prompt.

## Performance Notes

- **First query takes longer** due to model initialization
- **Subsequent queries are faster** (model cached in memory)
- **Vector search is O(1)** with ChromaDB
- **API handles concurrent requests** via FastAPI/Uvicorn

## Next Steps

- Add authentication (JWT tokens)
- Deploy to cloud (AWS, GCP, Azure)
- Add document management endpoints
- Implement caching for common queries
- Add conversation history/chat mode
- Create admin dashboard

## License

MIT

## Support

For issues or questions, check:

1. `.env` has valid `OPENAI_API_KEY`
2. Running from project root
3. Vector store is initialized (run `/ingest` first if needed)
4. Port 8000 is available
