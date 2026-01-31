# Agentic RAG System - Test Suite

Comprehensive automated testing suite for the Agentic RAG system, covering unit tests, integration tests, and CI/CD pipeline configuration.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing New Tests](#writing-new-tests)
- [CI/CD Pipeline](#cicd-pipeline)
- [Coverage](#coverage)

## Overview

This test suite ensures the reliability and correctness of the RAG system through:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **Mocked External Services**: Tests run without requiring actual API keys
- **Coverage Tracking**: Minimum 70% code coverage requirement
- **CI/CD Integration**: Automated testing on every push/PR

## Test Structure

```
tests/
├── README.md                      # This file
├── conftest.py                    # Shared fixtures and test configuration
├── pytest.ini                     # Pytest configuration
├── unit/                          # Unit tests
│   ├── ingestion/                 # Tests for data ingestion modules
│   │   ├── test_article_downloader.py
│   │   ├── test_chunker.py
│   │   ├── test_pdf_loader.py
│   │   ├── test_text_loader.py
│   │   └── test_yt_bot.py
│   ├── vectorstore/               # Tests for vector store modules
│   │   ├── test_chroma_store.py
│   │   └── test_embeddings.py
│   ├── generation/                # Tests for answer generation
│   │   └── test_answer_generator.py
│   └── utils/                     # Tests for utility modules
│       └── test_cost_tracker.py
└── integration/                   # Integration tests
    └── test_rag_pipeline.py       # End-to-end pipeline tests
```

## Running Tests

### Run All Tests

```bash
# Run all tests with verbose output
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=html --cov-report=term
```

### Run Specific Test Categories

```bash
# Run only unit tests
uv run pytest tests/unit -v -m unit

# Run only integration tests
uv run pytest tests/integration -v -m integration

# Run tests excluding slow tests
uv run pytest tests/ -v -m "not slow"

# Run only slow tests
uv run pytest tests/ -v -m slow
```

### Run Specific Test Files

```bash
# Run tests for a specific module
uv run pytest tests/unit/ingestion/test_chunker.py -v

# Run a specific test class
uv run pytest tests/unit/ingestion/test_chunker.py::TestChunker -v

# Run a specific test function
uv run pytest tests/unit/ingestion/test_chunker.py::TestChunker::test_chunk_single_document -v
```

### Run with Different Output Formats

```bash
# Short traceback
uv run pytest tests/ -v --tb=short

# Capture output
uv run pytest tests/ -v -s

# Show local variables on failure
uv run pytest tests/ -v -l

# Stop on first failure
uv run pytest tests/ -v -x
```

## Test Categories

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests for individual components
- `@pytest.mark.integration` - Integration tests for component interactions
- `@pytest.mark.slow` - Tests that take more than 5 seconds

### Unit Tests

Unit tests verify individual components in isolation:

#### Ingestion Tests
- **test_article_downloader.py**: Article downloading, HTML parsing, parallel processing
- **test_chunker.py**: Document chunking, size validation, metadata preservation
- **test_pdf_loader.py**: PDF extraction with PyMuPDF, text processing
- **test_text_loader.py**: Text file loading, metadata extraction
- **test_yt_bot.py**: YouTube video ID extraction, transcript processing

#### Vectorstore Tests
- **test_chroma_store.py**: ChromaDB operations, similarity search, metadata handling
- **test_embeddings.py**: OpenAI embeddings, batch processing, cost tracking

#### Generation Tests
- **test_answer_generator.py**: RAG answer generation, citation formatting, context handling

#### Utils Tests
- **test_cost_tracker.py**: API cost tracking, cost breakdown, history management

### Integration Tests

Integration tests verify complete workflows:

- **TestRAGPipeline**: End-to-end document processing workflows
- **TestParallelProcessing**: Ray-based parallelization tests
- **TestVectorStoreIntegration**: Vector store with embeddings
- **TestErrorHandling**: Error scenarios and edge cases
- **TestEndToEndScenarios**: Complete real-world scenarios

## Writing New Tests

### Test File Template

```python
"""
Unit tests for [module name].
"""

import pytest
from unittest.mock import Mock, patch
from src.module import function_to_test


@pytest.mark.unit
class TestModuleName:
    """Tests for [functionality description]."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = function_to_test()
        assert result is not None

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)

    @patch('external.dependency')
    def test_with_mock(self, mock_dependency):
        """Test with mocked dependency."""
        mock_dependency.return_value = "mocked"
        result = function_to_test()
        assert result == "expected"
```

### Using Fixtures

Common fixtures available in `conftest.py`:

```python
def test_with_fixtures(temp_dir, sample_documents, mock_openai_client):
    """Test using shared fixtures."""
    # temp_dir: Temporary directory (Path object)
    # sample_documents: List of sample Document objects
    # mock_openai_client: Mocked OpenAI client
    pass
```

### Best Practices

1. **Test One Thing**: Each test should verify one specific behavior
2. **Use Descriptive Names**: Test names should describe what they test
3. **Mock External Services**: Use mocks for API calls, file I/O
4. **Clean Up Resources**: Use fixtures with yield for cleanup
5. **Test Edge Cases**: Include tests for error conditions
6. **Keep Tests Fast**: Unit tests should run in milliseconds
7. **Document Complex Tests**: Add docstrings explaining test purpose

## CI/CD Pipeline

### GitHub Actions Workflow

Located at `.github/workflows/ci.yml`, the CI/CD pipeline runs:

1. **Test Job** (Python 3.11 & 3.12)
   - Install dependencies with uv
   - Run unit tests
   - Run integration tests (excluding slow)
   - Generate coverage reports
   - Upload to Codecov

2. **Lint Job**
   - Check code formatting with ruff
   - Verify import sorting

3. **Type Check Job**
   - Type checking with pyright

4. **Build Job**
   - Verify package imports
   - Confirm successful build

### Required Secrets

Configure these secrets in GitHub repository settings:

- `OPENAI_API_KEY`: OpenAI API key (for tests requiring it)
- `TAVILY_API_KEY`: Tavily API key (for tests requiring it)

Note: Most tests use mocks and don't require real API keys.

### Workflow Triggers

- Push to `main`, `dev`, or `feature/*` branches
- Pull requests to `main` or `dev` branches

## Coverage

### Coverage Requirements

- **Minimum Coverage**: 70% (enforced in pytest.ini)
- **Coverage Reports**: Generated in HTML and XML formats

### Viewing Coverage Reports

```bash
# Generate coverage report
uv run pytest tests/ --cov=src --cov-report=html

# Open HTML report
open htmlcov/index.html
```

### Coverage Reports Include

- Line coverage for all source files
- Missing line numbers
- Branch coverage
- Per-file and overall statistics

### Improving Coverage

1. Identify uncovered lines in HTML report
2. Write tests for uncovered code paths
3. Focus on critical business logic first
4. Add tests for error handling paths

## Continuous Improvement

### Adding New Tests

When adding new features:

1. Write tests **before** implementing feature (TDD)
2. Ensure tests fail before implementation
3. Implement feature until tests pass
4. Refactor with confidence

### Maintaining Tests

- Update tests when modifying code
- Remove obsolete tests
- Keep fixtures DRY (Don't Repeat Yourself)
- Regular review of test coverage

### Test Performance

- Keep unit tests under 100ms each
- Mark slow tests with `@pytest.mark.slow`
- Use mocks to avoid slow operations
- Profile slow tests and optimize

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure dependencies are installed
uv sync
```

**Fixture Not Found**
```bash
# Check conftest.py is in parent directory
# Verify fixture name spelling
```

**Slow Tests**
```bash
# Run without slow tests
uv run pytest -m "not slow"
```

**Coverage Not Generated**
```bash
# Install pytest-cov
uv add pytest-cov
```

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Contact

For questions or issues with tests, please open an issue in the repository.
