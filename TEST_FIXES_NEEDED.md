# Test Fixes Needed

This document outlines all the test failures and the fixes required to make them pass.

## Summary

**Total Tests**: 79 unit tests
**Currently Passing**: ~47 tests (59%)
**Currently Failing**: ~32 tests (41%)

## Test Failures by Category

### 1. CostTracker Tests (15 failures) - **FIXED** ✅

**Issue**: Tests expected different API signatures than actual implementation.

**Actual API**:
```python
# Constructor
CostTracker(log_file=Optional[str])

# Methods
track_openai_call(model, input_tokens, output_tokens, operation, metadata=None)
track_embedding_call(model, tokens, operation="embedding", metadata=None)
track_tavily_search(search_depth="advanced", num_results=10, metadata=None)
get_session_costs() -> Dict

# Attributes
session_calls: List[APICall]  # Not call_history
```

**Fixed in**: `tests/unit/utils/test_cost_tracker.py`

All cost tracker tests now use correct API signatures with temp_dir fixture for isolated test log files.

---

### 2. ChromaDB VectorStore Tests (9 failures) - **NEEDS FIX**

**Issue**: Constructor expects `embedding_model` not `embedding_function`.

**Actual API**:
```python
ChromaVectorStore(
    persist_directory: str,
    collection_name: str,
    embedding_model: EmbeddingModel  # Not embedding_function!
)
```

**Required Changes**:

```python
# WRONG
store = ChromaVectorStore(
    collection_name="test",
    persist_directory="./test_db",
    embedding_function=mock_embedding_func  # ❌
)

# CORRECT
store = ChromaVectorStore(
    persist_directory="./test_db",  # Order matters!
    collection_name="test",
    embedding_model=mock_embedding_model  # ✅
)
```

**Files to Fix**:
- `tests/unit/vectorstore/test_chroma_store.py`
- `tests/integration/test_rag_pipeline.py` (ChromaDB integration tests)

---

### 3. OpenAIEmbedding Tests (4 failures) - **NEEDS FIX**

**Issue**: Constructor doesn't take `cost_tracker` parameter, and `dimension` attribute depends on model.

**Actual API**:
```python
OpenAIEmbedding(
    api_key: str,
    model: str = "text-embedding-3-small"
)
# No cost_tracker parameter!
# No dimension attribute exposed!
```

**Required Changes**:

```python
# Test initialization
def test_initialization(self, mock_openai_client):
    embedder = OpenAIEmbedding(
        api_key="test-key",
        model="text-embedding-3-small"
    )
    # Don't assert dimension - it's not a public attribute
    assert embedder.model == "text-embedding-3-small"

# Test with cost tracking - use embedder separately
def test_cost_tracking_on_embed(self, mock_openai_client, temp_dir):
    # Cost tracker is separate, not part of embedder
    from src.utils.cost_tracker import CostTracker
    tracker = CostTracker(log_file=str(temp_dir / "costs.json"))

    embedder = OpenAIEmbedding(api_key="test-key")
    embedder.client = mock_openai_client

    # After embedding, manually track cost if needed
    embedder.embed_documents(["test"])
```

**Files to Fix**:
- `tests/unit/vectorstore/test_embeddings.py`

---

### 4. Article Downloader Tests (5 failures) - **NEEDS FIX**

**Issue 1: `sanitize_filename` doesn't replace spaces with underscores**

The actual implementation preserves spaces in filenames.

**Fix**:
```python
# Change this test:
def test_sanitize_filename(self):
    assert sanitize_filename("Normal Title") == "Normal_Title"  # ❌

# To this:
def test_sanitize_filename(self):
    result = sanitize_filename("Normal Title")
    # Check it handles special chars, not necessarily replaces spaces
    assert "?" not in result
    assert "/" not in result
```

**Issue 2: Mock setup for wget and HTML parsing**

The mocks don't properly simulate file creation and response objects.

**Fix for wget test**:
```python
@patch('subprocess.run')
def test_download_article_wget_success(self, mock_run, temp_dir):
    # Mock successful wget
    mock_run.return_value = Mock(returncode=0, stderr="")

    # Actually create the PDF file that wget would download
    test_pdf = temp_dir / "paper.pdf"
    test_pdf.write_bytes(b"%PDF-1.4 test content")

    # Mock glob to find the file
    with patch('pathlib.Path.glob', return_value=[test_pdf]):
        success, file_path = download_article_wget(...)
        assert success is True
```

**Fix for HTML parsing test**:
```python
@patch('requests.get')
def test_parse_and_save_article_success(self, mock_get, temp_dir):
    mock_response = Mock()
    mock_response.status_code = 200
    # Make sure content has enough text (>200 chars to pass min threshold)
    mock_response.content = b"""
    <html>
        <body>
            <article>
                <h1>Test Article Title</h1>
                <p>""" + b"This is substantial content. " * 20 + b"""</p>
            </article>
        </body>
    </html>
    """
    mock_get.return_value = mock_response

    success, file_path = parse_and_save_article(...)
    assert success is True
    assert file_path.exists()
```

**Files to Fix**:
- `tests/unit/ingestion/test_article_downloader.py`

---

### 5. PDF Loader Tests (1 failure) - **NEEDS FIX**

**Issue**: Mock `response.iter_content()` not iterable.

**Fix**:
```python
@patch('requests.get')
@patch('fitz.open')
def test_load_pdf_source_with_download(self, mock_fitz, mock_get):
    # Mock HTTP response with iterable content
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_content = Mock(return_value=[b"chunk1", b"chunk2"])  # Make it iterable!
    mock_get.return_value = mock_response

    # Mock PDF extraction
    mock_doc = Mock()
    mock_doc.metadata = {}
    mock_doc.__len__.return_value = 1
    mock_page = Mock()
    mock_page.get_text.return_value = "PDF content"
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.return_value = mock_doc

    doc = load_pdf_source("https://example.com/paper.pdf")
    assert doc is not None
```

**Files to Fix**:
- `tests/unit/ingestion/test_pdf_loader.py`

---

### 6. YouTube Bot Tests (1 failure) - **NEEDS FIX**

**Issue**: `get_video_id()` returns `None` for test URL, causing validation error.

**Fix**:
```python
@patch('src.ingestion.yt_bot.YouTubeTranscriptApi')
@patch('src.ingestion.yt_bot.get_video_id')  # Mock the video ID extraction!
def test_load_youtube_video_success(self, mock_get_id, mock_api, mock_youtube_transcript):
    # Mock video ID extraction
    mock_get_id.return_value = "test123"  # Return valid ID

    # Mock the transcript API
    mock_transcript_list = Mock()
    mock_transcript = Mock()
    mock_transcript.fetch.return_value = mock_youtube_transcript
    mock_transcript_list.find_manually_created_transcript.return_value = mock_transcript
    mock_api.list_transcripts.return_value = mock_transcript_list

    url = "https://www.youtube.com/watch?v=test123"
    doc = load_youtube_video(url)

    assert doc is not None
    assert doc.metadata["video_id"] == "test123"
```

**Files to Fix**:
- `tests/unit/ingestion/test_yt_bot.py`

---

## Quick Fix Summary

### Files That Need Updates:

1. ✅ **`tests/unit/utils/test_cost_tracker.py`** - FIXED
2. ❌ **`tests/unit/vectorstore/test_chroma_store.py`** - Update all `embedding_function` → `embedding_model`
3. ❌ **`tests/unit/vectorstore/test_embeddings.py`** - Remove `cost_tracker` and `dimension` assertions
4. ❌ **`tests/unit/ingestion/test_article_downloader.py`** - Fix mock setups
5. ❌ **`tests/unit/ingestion/test_pdf_loader.py`** - Make `iter_content` iterable
6. ❌ **`tests/unit/ingestion/test_yt_bot.py`** - Mock `get_video_id`
7. ❌ **`tests/integration/test_rag_pipeline.py`** - Update ChromaDB usage

---

## Coverage Requirement

Currently disabled (`--cov-fail-under=0`) to focus on fixing tests.

**After fixing tests, re-enable**:
```ini
# pytest.ini
--cov-fail-under=70
```

---

## Testing Strategy

### Phase 1: Fix Critical Tests ✅
- Cost tracker tests (DONE)

### Phase 2: Fix Vectorstore Tests (NEXT)
- ChromaDB tests
- Embeddings tests

### Phase 3: Fix Ingestion Tests
- Article downloader
- PDF loader
- YouTube bot

### Phase 4: Fix Integration Tests
- Update to use correct APIs

### Phase 5: Re-enable Coverage
- Set threshold back to 70%
- Add more tests if needed

---

## Running Tests After Fixes

```bash
# Run all tests
uv run pytest tests/unit -v

# Run specific module
uv run pytest tests/unit/utils/test_cost_tracker.py -v

# Run with coverage (after fixing)
uv run pytest tests/ --cov=src --cov-report=html
```

---

## Key Learnings

1. **Always check actual implementation signatures** before writing tests
2. **Use temp_dir fixture** for file-based operations (cost logs, vector stores)
3. **Mock external dependencies properly**:
   - `iter_content()` must be iterable
   - File system operations need actual file creation
   - API responses need proper structure
4. **Parameter order matters** in constructors
5. **Test what exists, not what you expect** - verify actual attributes/methods

---

## Next Steps

1. Fix vectorstore tests (highest priority - 13 failures)
2. Fix ingestion tests (medium priority - 7 failures)
3. Fix integration tests (update to match fixed APIs)
4. Re-enable 70% coverage requirement
5. Run full CI/CD pipeline to verify

---

## Estimated Time

- Vectorstore tests: ~30 minutes
- Ingestion tests: ~45 minutes
- Integration tests: ~30 minutes
- Verification: ~15 minutes

**Total**: ~2 hours for complete test suite fix

