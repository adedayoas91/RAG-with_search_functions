# Document Sources - User Guide

The Agentic RAG System now supports flexible document sourcing with three modes of operation:

## 📌 Document Source Modes

### 1. 🌐 Online Search (Default)
Fetches documents from the web using Tavily search.

**Best for:**
- Researching current topics
- Accessing public knowledge bases
- Finding academic papers and articles
- Discovering YouTube educational content

**How it works:**
1. Enter your research query
2. System searches up to 100 online sources
3. Sources are filtered by relevance
4. You approve sources to use
5. Documents are downloaded/parsed automatically

**Example:**
```
What would you like to research? Latest developments in quantum computing

How would you like to gather documents for your research?
  1. 🌐 Search online (web articles, PDFs, YouTube videos)
  2. 📁 Upload local documents (PDFs, text files)
  3. 🔄 Both (combine online search + local documents)

Enter your choice (1/2/3): 1
```

---

### 2. 📁 Local Document Upload
Loads documents from your local filesystem.

**Best for:**
- Analyzing your own documents
- Working with proprietary/private data
- Offline research
- Corporate knowledge base integration
- Personal notes and papers

**Supported formats:**
- PDF files (`.pdf`)
- Text files (`.txt`)
- Markdown files (`.md`)

**How it works:**
1. Enter your research query
2. Choose local document mode
3. Provide path to your documents directory
4. Choose whether to scan subdirectories
5. Review and confirm documents to load
6. System loads all documents automatically

**Example:**
```
What would you like to research? Key insights from my research papers

How would you like to gather documents for your research?
  1. 🌐 Search online (web articles, PDFs, YouTube videos)
  2. 📁 Upload local documents (PDFs, text files)
  3. 🔄 Both (combine online search + local documents)

Enter your choice (1/2/3): 2

📁 Local Document Upload
============================================================
Please provide the path to your documents directory.
Supported formats: PDF (.pdf), Text (.txt)

Enter path to documents directory: ~/Documents/research_papers

Scan subdirectories? (y/n, default: y): y

📄 Found 15 documents:
  1. paper1.pdf (523.4 KB)
  2. paper2.pdf (789.2 KB)
  3. notes.txt (12.5 KB)
  ...

Load all 15 documents? (y/n): y
```

---

### 3. 🔄 Both (Hybrid Mode)
Combines online search results with local documents.

**Best for:**
- Comprehensive research combining multiple sources
- Validating online information with personal knowledge
- Augmenting proprietary data with public sources
- Building complete knowledge bases

**How it works:**
1. Enter your research query
2. Choose "both" mode
3. System performs online search (same as mode 1)
4. You approve online sources
5. System prompts for local documents directory
6. All documents are combined for analysis

**Example:**
```
What would you like to research? Compare machine learning frameworks

How would you like to gather documents for your research?
  1. 🌐 Search online (web articles, PDFs, YouTube videos)
  2. 📁 Upload local documents (PDFs, text files)
  3. 🔄 Both (combine online search + local documents)

Enter your choice (1/2/3): 3

[Online search process...]
✅ Successfully loaded 12 online sources

[Local document loading process...]
✅ Successfully loaded 8 local documents

📊 Combined Documents Summary:
  • Total documents: 20
  • Online sources: 12
  • Local documents: 8
```

---

## 📂 Organizing Local Documents

### Directory Structure
The system can scan both flat and nested directory structures:

```
my_documents/
├── research_papers/
│   ├── ml_paper1.pdf
│   ├── ml_paper2.pdf
│   └── notes.txt
├── company_docs/
│   ├── strategy.pdf
│   └── analysis.md
└── personal_notes.txt
```

### Best Practices

1. **Organize by topic**: Keep related documents in subdirectories
2. **Use descriptive names**: File names help identify content
3. **Clean up**: Remove irrelevant documents before scanning
4. **Check formats**: Ensure documents are in supported formats (.pdf, .txt, .md)
5. **Size considerations**: Very large documents may take longer to process

---

## 🔍 Feature Comparison

| Feature | Online Search | Local Upload | Both |
|---------|--------------|--------------|------|
| Internet required | ✅ Yes | ❌ No | ✅ Yes |
| API costs | ✅ Yes (search) | ❌ No | ✅ Yes (search) |
| Private documents | ❌ No | ✅ Yes | ✅ Yes |
| Public sources | ✅ Yes | ❌ No | ✅ Yes |
| Source approval | ✅ Yes | ✅ Auto-approve | ✅ Yes (online) |
| Speed | Medium | Fast | Slower |

---

## 💡 Use Cases

### Academic Research
**Mode: Both**
- Online: Latest papers from arXiv, Google Scholar
- Local: Your own research notes and unpublished work

### Corporate Knowledge Base
**Mode: Local**
- Load company documents, policies, reports
- Keep data private and secure

### Content Creation
**Mode: Online**
- Gather diverse sources from the web
- Research current trends and topics

### Personal Learning
**Mode: Both**
- Online: Educational videos, tutorials
- Local: Your notes, textbooks, saved articles

---

## 🚀 Getting Started

### Quick Start with Example Documents

1. Use the provided example documents:
   ```bash
   # Example directory is already created at ./example_documents/
   ```

2. Run the RAG system:
   ```bash
   uv run main.py
   ```

3. Choose option 2 (Local documents)

4. Enter path: `./example_documents`

5. Confirm loading the sample document

### Adding Your Own Documents

1. Create a documents directory:
   ```bash
   mkdir ~/my_research_docs
   ```

2. Add your PDFs and text files:
   ```bash
   cp your_document.pdf ~/my_research_docs/
   ```

3. Run the RAG system and choose local mode

4. Enter your directory path when prompted

---

## ⚠️ Important Notes

### Document Processing
- **PDFs**: Automatically extracted using PyMuPDF
- **Text files**: Loaded directly with UTF-8 encoding
- **Large files**: May take longer to process and embed

### Privacy & Security
- Local documents never leave your machine
- Only metadata (file names, sizes) are displayed
- Online searches use Tavily API (respects their privacy policy)

### Performance Tips
- **Small document sets** (< 50 files): Process all at once
- **Large document sets** (> 100 files): Consider filtering by subdirectory
- **Very large PDFs** (> 50MB): May want to process separately

### Troubleshooting

**Issue**: "No supported documents found"
- Check file extensions (.pdf, .txt, .md only)
- Verify directory path is correct
- Ensure you have read permissions

**Issue**: "Failed to load document"
- Check if PDF is corrupted or encrypted
- Verify text file encoding (UTF-8 recommended)
- Check file isn't open in another program

**Issue**: "Path does not exist"
- Use absolute path or path with ~
- Check for typos in path
- Verify directory exists

---

## 📝 Example Workflows

### Workflow 1: Research Paper Analysis
```
1. Query: "Summarize key findings from my papers"
2. Mode: Local (option 2)
3. Path: ~/research/papers/
4. Let system load all PDFs
5. Get comprehensive summary with citations
```

### Workflow 2: Market Research
```
1. Query: "Current trends in electric vehicles"
2. Mode: Both (option 3)
3. Online: Approve 10-15 recent articles
4. Local: Load your market analysis reports
5. Get insights combining public + private data
```

### Workflow 3: Learning Topic
```
1. Query: "Explain neural networks"
2. Mode: Online (option 1)
3. Approve mix of articles, papers, videos
4. Get comprehensive explanation from diverse sources
```

---

## 🔧 Technical Details

### File Loading Process
1. **Scan**: Directory traversal to find supported files
2. **Display**: Show file list with sizes for user confirmation
3. **Load**: Parse each file using appropriate loader
4. **Extract**: Extract text content and metadata
5. **Document**: Create LangChain Document objects

### Metadata Preserved
- File path (source)
- File type (source_type)
- File name (title)
- Original location (for local files)

### Integration with Pipeline
Local documents are treated identically to online sources after loading:
- Same chunking process
- Same embedding generation
- Same vector storage
- Same retrieval mechanism

---

## 📚 Additional Resources

- See `src/ingestion/local_document_loader.py` for implementation
- Check `main.py` for integration details
- Review `config.py` for configuration options

---

## 🆕 Future Enhancements

Planned features for local document loading:
- Support for more formats (DOCX, HTML, Markdown)
- Selective file loading (choose specific files)
- Document metadata editing before loading
- Batch processing for very large collections
- Watch folder for automatic updates
