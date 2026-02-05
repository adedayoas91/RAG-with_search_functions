# RAG System with Search v1.2

A production-ready Retrieval-Augmented Generation (RAG) system with flexible document sourcing, parallel processing, and a comprehensive testing infrastructure. This system is designed to answer questions based on a given context, which it can source from the web or local files.

**Latest Updates:**
- ✅ **LangChain 1.x compatibility** - Updated to modern LangChain APIs
- ✅ **95 passing tests** - Comprehensive test coverage across all modules
- ✅ **GitHub Actions CI/CD** - Automated testing and quality checks
- ✅ **UV package manager** - Fast, reliable dependency management

## ✨ Key Features

### 🔄 Flexible Document Sources
- **Online Search**: Fetch up to 100 sources from the web (articles, PDFs, YouTube).
- **Local Upload**: Load documents from your local filesystem.
- **Hybrid Mode**: Combine both online and local sources.

### 🚀 Performance Optimized
- Parallel document downloading (5 workers).
- Ray-based parallel chunking (4 workers).
- Distributed embedding generation.
- Efficient vector storage with ChromaDB.

### 📊 Advanced Features
- Source filtering by relevance.
- Interactive source approval.
- Numeric citations [1], [2,3], [2-5].
- Real-time cost tracking.
- Session analytics.

### 🧪 Production Ready
- Comprehensive test suite (95 tests - 80 unit, 15 integration).
- GitHub Actions CI/CD pipeline with automated testing.
- 46% code coverage (continuously improving).
- Type checking with Pyright and linting with Ruff.

## 🔄 System Workflow

```
┌─────────────────┐
│   USER QUERY    │
│  "What is AI?"  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     ┌─────────────────┐
│ DOCUMENT SOURCE │────▶│  MODE SELECTION │
│   SELECTION     │     │ • 🌐 Online     │
│ • Online        │     │ • 📁 Local      │
│ • Local         │     │ • 🔄 Both       │
│ • Both          │     └─────────┬───────┘
└─────────────────┘               │
                                  ▼
                    ┌─────────────────────┐
                    │  DOCUMENT GATHERING │
                    │                     │
                    │ ┌─────────┬─────────┐ │
                    │ │ ONLINE  │ LOCAL   │ │
                    │ ├─────────┼─────────┤ │
                    │ │ Tavily  │ File    │ │
                    │ │ Search  │ Scan    │ │
                    │ │ Filter  │ Load    │ │
                    │ │ Download│         │ │
                    │ └─────────┴─────────┘ │
                    └─────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────┐
                    │  DOCUMENT PROCESSING │
                    │                     │
                    │ ┌─────────┬─────────┐ │
│ │ CHUNK   │ EMBED   │ │
│ ├─────────┼─────────┤ │
│ │ Split   │ Vector  │ │
│ │ Text    │ Store   │ │
│ │ (80)    │ (Chroma)│ │
                    │ └─────────┴─────────┘ │
                    └─────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────┐
                    │    RETRIEVAL        │
                    │                     │
                    │ • Similarity Search │
                    │ • Top-K Results     │
                    │ • Context Assembly  │
                    │                     │
                    └─────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────┐
                    │   GENERATION        │
                    │                     │
                    │ • GPT-4 Answer      │
                    │ • Citation Links    │
                    │ • Cost Tracking     │
                    │                     │
                    └─────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────┐
                    │   FINAL RESPONSE    │
                    │                     │
                    │ "AI is... [1][2][3]"│
                    │                     │
                    │ Sources: [1] paper.pdf │
                    │         [2] article.html │
                    │         [3] research.pdf │
                    └─────────────────────┘
```

### Workflow Steps:

1. **Query Input** → User provides research question
2. **Source Selection** → Choose online, local, or both document sources
3. **Document Gathering** → Collect relevant documents based on selection
4. **Processing Pipeline** → Chunk text → Generate embeddings → Store in vector DB
5. **Retrieval** → Find most relevant document chunks using similarity search
6. **Generation** → Synthesize answer with citations using retrieved context
7. **Response** → Deliver formatted answer with source references

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Document Source Modes](#document-source-modes)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [API Keys](#api-keys)
- [Contributing](#contributing)

## 🔧 Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd rag-with_search

# Install dependencies with uv
uv sync

# Create .env file with your API keys
echo "OPENAI_API_KEY=your_openai_api_key" > .env
echo "TAVILY_API_KEY=your_tavily_api_key" >> .env
```

### Required API Keys

Add these to your `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 🚀 Quick Start

### Basic Usage

```bash
uv run main.py
```

### Example Session

```
What would you like to research? Machine learning fundamentals

How would you like to gather documents for your research?
  1. 🌐 Search online (web articles, PDFs, YouTube videos)
  2. 📁 Upload local documents (PDFs, text files)
  3. 🔄 Both (combine online search + local documents)

Enter your choice (1/2/3): 1

[System searches, filters, and presents sources...]
[You approve sources...]
[System processes and generates answer with citations...]
```

## 📚 Document Source Modes

### 1. Online Search Mode

Search and retrieve documents from the web.

**Features:**
- Tavily search (up to 100 sources)
- Relevance filtering
- Source summarization
- User approval workflow
- Automatic download/parsing

**Supported sources:**
- Web articles (parsed HTML)
- PDF documents (downloaded)
- YouTube videos (transcripts)

### 2. Local Upload Mode

Load documents from your local filesystem.

**Features:**
- Directory scanning (recursive optional)
- File preview before loading
- Automatic format detection
- No internet required

**Supported formats:**
- PDF files (`.pdf`)
- Text files (`.txt`)
- Markdown files (`.md`)

**Example:**
```bash
# Create a documents directory
mkdir my_documents
cp *.pdf my_documents/

# Run the system and choose option 2
uv run main.py
# Enter: my_documents/
```

### 3. Hybrid Mode (Both)

Combine online search with local documents.

**Features:**
- All online search features
- All local upload features
- Combined document analysis
- Comprehensive knowledge base

**Best for:**
- Academic research (online papers + personal notes)
- Corporate knowledge (public data + internal docs)
- Validation (cross-reference sources)

See [DOCUMENT_SOURCES.md](DOCUMENT_SOURCES.md) for detailed documentation.

## ⚙️ Configuration

### Edit `config.py`

```python
# Search settings
max_results = 100
relevance_threshold = 0.7

# Chunking
chunk_size = 80
chunk_overlap = 20

# Retrieval
retrieval_k = 10

# Models
generation_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"
```

### Environment Variables

```env
# Required
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Optional
CHUNK_SIZE=1000
RETRIEVAL_K=10
```

## 💡 Usage Examples

### Example 1: Research with Online Sources

```bash
uv run main.py
```
```
Query: Latest developments in quantum computing
Mode: 1 (Online)
→ System fetches 100 sources
→ Filters to 30 relevant sources
→ You approve 15 sources
→ Downloads/parses articles
→ Generates comprehensive answer
```

### Example 2: Analyze Local Documents

```bash
uv run main.py
```
```
Query: Summarize key findings from my research papers
Mode: 2 (Local)
Path: ~/Documents/research/
→ Scans directory
→ Finds 20 PDFs
→ Loads all documents
→ Generates summary with citations
```

### Example 3: Hybrid Research

```bash
uv run main.py
```
```
Query: Compare machine learning frameworks
Mode: 3 (Both)
→ Searches online (12 sources approved)
→ Loads local docs (8 PDFs)
→ Analyzes 20 combined documents
→ Generates comprehensive comparison
```

### Example 4: Try the Sample Documents

```bash
uv run main.py
```
```
Query: What features does the RAG system support?
Mode: 2 (Local)
Path: ./example_documents
→ Loads sample_document.txt
→ Generates answer about system features
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run unit tests only
uv run python -m pytest tests/unit -v

# Run integration tests only
uv run python -m pytest tests/integration -v

# Run with coverage
uv run python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

### Test Suite

- **95 total tests** (all passing ✅)
  - 80 unit tests
  - 15 integration tests
- **Test coverage**: 46% (improving toward 70% target)
- **CI/CD**: GitHub Actions on every push/PR to main/dev branches

See [tests/README.md](tests/README.md) for detailed testing documentation.

## 📁 Project Structure

```
RAG-with_search_functions/
├── main.py                      # Main entry point
├── config.py                    # Configuration
├── .env                         # API keys (create from .env.example)
├── README.md                    # This file
├── DOCUMENT_SOURCES.md         # Document source modes guide
├── TESTING_SUMMARY.md          # Testing infrastructure overview
│
├── src/
│   ├── ingestion/              # Document loading and processing
│   │   ├── web_search.py       # Tavily search client
│   │   ├── local_document_loader.py  # Local file loading
│   │   ├── article_downloader.py     # Parallel downloading
│   │   ├── pdf_loader.py       # PDF extraction
│   │   ├── text_loader.py      # Text file loading
│   │   ├── yt_bot.py           # YouTube transcripts
│   │   ├── chunker.py          # Document chunking
│   │   ├── google_search.py    # Google search integration
│   │   ├── source_filter.py    # Source relevance filtering
│   │   ├── source_summarizer.py # Source summarization
│   │   └── article_loader.py   # Article processing
│   │
│   ├── generation/             # Answer generation
│   │   ├── agent.py            # LangChain agent setup
│   │   ├── answer_generator.py # RAG answer generation
│   │   ├── generate.py         # Generation pipeline
│   │   └── tools.py            # Agent tools
│   │
│   ├── search/                 # Search functionality
│   │   ├── search_agent.py     # Search agent
│   │   └── search_tools.py     # Search tools
│   │
│   ├── vectorstore/            # Embeddings and vector storage
│   │   ├── embeddings.py       # OpenAI embeddings
│   │   ├── chroma_store.py     # ChromaDB integration
│   │   └── parallel_embedding.py  # Ray parallelization
│   │
│   └── utils/                  # Utilities
│       ├── cost_tracker.py     # API cost tracking
│       ├── cli_display.py      # User interface
│       ├── data_persistence.py # Data persistence
│       └── logging_config.py   # Logging setup
│
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests (79 tests)
│   ├── integration/            # Integration tests (17 tests)
│   ├── conftest.py            # Shared fixtures
│   └── README.md              # Testing documentation
│
├── example_documents/          # Sample documents for testing
│   └── sample_document.txt
│
├── data/                       # Data directory (created at runtime)
│   ├── downloads/             # Downloaded articles
│   ├── chroma_db/            # Vector database
│   └── analytics.json        # Session analytics
│
└── .github/
    └── workflows/
        └── ci.yml             # GitHub Actions CI/CD
```

## 🔑 API Keys

### OpenAI API Key

1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Navigate to API Keys
3. Create new secret key
4. Add to `.env` file

**Cost estimates:**
- Embeddings: ~$0.0001 per query
- Generation: ~$0.0006 per 1K output tokens
- Typical session: $0.10-$0.30

### Tavily API Key

1. Sign up at [tavily.com](https://tavily.com)
2. Get API key from dashboard
3. Add to `.env` file

**Features:**
- 100 searches per month (free tier)
- Advanced search depth
- Paywall detection
- Source metadata

## 💰 Cost Tracking

The system tracks API costs in real-time:

- OpenAI API calls (embeddings + generation)
- Tavily search calls
- Per-session breakdown
- Historical analytics

View costs at end of each session or in `data/cost_log.json`.

## 🚦 CI/CD Pipeline

### GitHub Actions Workflow ✅

- **Test Job**: Runs on Python 3.12 with UV package manager
- **Lint Job**: Code formatting with Ruff
- **Type Check Job**: Static type analysis with Pyright
- **Build Job**: Package verification and import testing

### Current Status
- ✅ **All tests passing** (95/95)
- ✅ **Coverage reports** generated automatically
- ✅ **Multi-stage pipeline** with proper dependencies

### Triggers

- Push to `main`, `dev` branches
- Pull requests to `main`, `dev` branches

See [.github/workflows/ci.yml](.github/workflows/ci.yml) for configuration.

## 🐛 Troubleshooting

### Common Issues

**"No module named 'src'"**
```bash
# Run from project root with uv
uv run main.py
```

**"Missing API keys"**
```bash
# Check .env file exists and has keys
cat .env
```

**"Ray failed to start"**
```bash
# Check Ray is installed
uv sync
```

**"No supported documents found"** (Local mode)
```bash
# Check file extensions and path
ls -R your_documents_directory/
```

## 🤝 Contributing

### Development Setup

```bash
# Install dev dependencies
uv sync

# Run tests
uv run python -m pytest tests/ -v

# Run linting
uv run ruff check src/ tests/

# Run type checking
uv run pyright src/
```

### Adding Features

1. Create feature branch
2. Write tests first (TDD)
3. Implement feature
4. Ensure tests pass
5. Submit pull request

### Code Style

- Follow PEP8
- Add type hints
- Write docstrings
- Keep functions focused

## 📜 License

[Add your license here]

## 🙏 Acknowledgments

- **OpenAI** for GPT models and embeddings API
- **Tavily** for advanced web search API
- **ChromaDB** for efficient vector storage
- **Ray** for distributed processing and parallelization
- **LangChain 1.x** for modern RAG framework and document processing
- **UV** for fast, reliable Python package management
- **GitHub Actions** for CI/CD automation

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review test examples
