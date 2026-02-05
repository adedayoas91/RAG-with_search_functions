"""
Local document loader for user-uploaded documents.

This module handles loading documents from the user's local filesystem,
supporting PDF, TXT, DOCX, and other common document formats.
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

from src.utils import get_logger
from src.ingestion.pdf_loader import load_pdf_from_file
from src.ingestion.text_loader import load_text_file

logger = get_logger(__name__)


def get_user_upload_documents(query: str = "user_uploads") -> List[Document]:
    """
    Handle user document uploads by creating an upload directory
    and allowing user to add files before loading.

    Args:
        query: Query string to create directory name

    Returns:
        List of loaded Document objects
    """
    print("\n📁 User Document Upload")
    print("=" * 60)
    print("I'll create an upload directory where you can add your documents.")
    print("Supported formats: PDF (.pdf), Text (.txt)")
    print()

    # Create user upload directory in data/downloads (same as downloads)
    from src.ingestion.article_downloader import sanitize_filename
    dir_name = sanitize_filename(query.lower().replace(' ', '_'))[:50]
    upload_dir = Path("./data/downloads") / dir_name
    upload_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ Created upload directory: {upload_dir.absolute()}")
    print()
    print("📋 Instructions:")
    print("1. Open a new terminal/file explorer window")
    print(f"2. Copy your documents to: {upload_dir.absolute()}")
    print("3. Return here and press Enter when ready")
    print()
    print("💡 Tip: Supported formats are PDF and TXT files")
    print()

    # Wait for user to add files
    input("Press Enter when you've added your documents: ").strip()

    # Check if any files were added
    document_files = scan_directory_for_documents(upload_dir, recursive=True)

    if not document_files:
        print(f"⚠️  No supported documents found in {upload_dir}")
        print("Please add PDF or TXT files to the directory and try again.")
        return []

    print(f"\n📄 Found {len(document_files)} documents:")
    for i, file_path in enumerate(document_files, 1):
        file_size = file_path.stat().st_size / 1024  # KB
        print(f"  {i}. {file_path.name} ({file_size:.1f} KB)")

    # Confirm loading
    print()
    confirm = input(f"Load all {len(document_files)} documents? (y/n): ").strip().lower()

    if confirm != 'y':
        print("❌ Document loading cancelled")
        return []

    # Load documents
    print(f"\n📥 Loading documents...")
    documents = []
    failed_count = 0

    for idx, file_path in enumerate(document_files, 1):
        print(f"  [{idx}/{len(document_files)}] Loading: {file_path.name}...", end=" ")

        try:
            if file_path.suffix.lower() == '.pdf':
                doc = load_pdf_from_file(
                    str(file_path),
                    source_url=f"file://{file_path.absolute()}"
                )
            elif file_path.suffix.lower() == '.txt':
                doc = load_text_file(
                    str(file_path),
                    source_url=f"file://{file_path.absolute()}"
                )
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            documents.append(doc)
            print("✅")

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {str(e)}")
            print("❌")
            failed_count += 1
            continue

    success_count = len(documents)
    if success_count > 0:
        print(f"\n✅ Successfully loaded {success_count} documents")
        if failed_count > 0:
            print(f"⚠️  Failed to load {failed_count} documents")

    return documents


def get_local_documents_path() -> Optional[Path]:
    """
    Prompt user for path to local documents directory.

    Returns:
        Path to documents directory, or None if invalid
    """
    print("\n📁 Local Document Upload")
    print("=" * 60)
    print("Please provide the path to your documents directory.")
    print("Supported formats: PDF (.pdf), Text (.txt)")
    print()

    path_input = input("Enter path to documents directory: ").strip()

    if not path_input:
        logger.warning("No path provided")
        return None

    # Expand ~ and resolve path
    doc_path = Path(path_input).expanduser().resolve()

    if not doc_path.exists():
        logger.error(f"Path does not exist: {doc_path}")
        print(f"❌ Path does not exist: {doc_path}")
        return None

    if not doc_path.is_dir():
        logger.error(f"Path is not a directory: {doc_path}")
        print(f"❌ Path is not a directory: {doc_path}")
        return None

    return doc_path


def scan_directory_for_documents(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Scan directory for supported document files.

    Args:
        directory: Path to directory to scan
        recursive: Whether to scan subdirectories

    Returns:
        List of document file paths
    """
    supported_extensions = {'.pdf', '.txt', '.md'}
    document_files = []

    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in supported_extensions:
                    document_files.append(file_path)
    else:
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                document_files.append(file_path)

    return sorted(document_files)


def load_local_document(file_path: Path) -> Optional[Document]:
    """
    Load a single local document file.

    Args:
        file_path: Path to document file

    Returns:
        Document object, or None if loading failed
    """
    try:
        file_ext = file_path.suffix.lower()

        if file_ext == '.pdf':
            logger.info(f"Loading PDF: {file_path.name}")
            return load_pdf_from_file(
                str(file_path),
                source_url=str(file_path)
            )

        elif file_ext in {'.txt', '.md'}:
            logger.info(f"Loading text file: {file_path.name}")
            return load_text_file(
                str(file_path),
                source_url=str(file_path)
            )

        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return None

    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {str(e)}")
        return None


def load_local_documents(directory_path: Optional[Path] = None) -> List[Document]:
    """
    Load all supported documents from a local directory.

    Args:
        directory_path: Path to documents directory (prompts user if None)

    Returns:
        List of loaded Document objects
    """
    if directory_path is None:
        directory_path = get_local_documents_path()
        if directory_path is None:
            return []

    logger.info(f"Scanning directory: {directory_path}")

    # Ask user about recursive scanning
    recursive_input = input("Scan subdirectories? (y/n, default: y): ").strip().lower()
    recursive = recursive_input != 'n'

    # Scan for documents
    document_files = scan_directory_for_documents(directory_path, recursive=recursive)

    if not document_files:
        logger.warning("No supported documents found in directory")
        print(f"⚠️  No supported documents found in {directory_path}")
        return []

    print(f"\n📄 Found {len(document_files)} documents:")
    for i, file_path in enumerate(document_files, 1):
        rel_path = file_path.relative_to(directory_path)
        file_size = file_path.stat().st_size / 1024  # KB
        print(f"  {i}. {rel_path} ({file_size:.1f} KB)")

    # Ask user to confirm
    print()
    confirm = input(f"Load all {len(document_files)} documents? (y/n): ").strip().lower()

    if confirm != 'y':
        print("❌ Document loading cancelled")
        return []

    # Load documents
    print(f"\n📥 Loading documents...")
    documents = []
    failed_count = 0

    for idx, file_path in enumerate(document_files, 1):
        print(f"  [{idx}/{len(document_files)}] Loading: {file_path.name}...", end=" ")

        doc = load_local_document(file_path)

        if doc is not None:
            documents.append(doc)
            print("✓")
        else:
            failed_count += 1
            print("✗")

    logger.info(f"Successfully loaded {len(documents)}/{len(document_files)} documents")

    if documents:
        print(f"\n✅ Successfully loaded {len(documents)} documents")
        if failed_count > 0:
            print(f"⚠️  Failed to load {failed_count} documents")
    else:
        print(f"❌ Failed to load any documents")

    return documents


def get_document_source_mode() -> str:
    """
    Prompt user to choose document source mode.

    Returns:
        'online', 'local', or 'both'
    """
    print("\n🔍 Document Source Selection")
    print("=" * 60)
    print("How would you like to gather documents for your research?")
    print()
    print("  1. 🌐 Search online (web articles, PDFs, YouTube videos)")
    print("  2. 📁 Upload local documents (PDFs, text files)")
    print("  3. 🔄 Both (combine online search + local documents)")
    print()

    while True:
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == '1':
            logger.info("User selected: online search")
            return 'online'
        elif choice == '2':
            logger.info("User selected: local documents")
            return 'local'
        elif choice == '3':
            logger.info("User selected: both online and local")
            return 'both'
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def print_document_summary(documents: List[Document], source_type: str):
    """
    Print summary of loaded documents.

    Args:
        documents: List of loaded documents
        source_type: 'online' or 'local'
    """
    if not documents:
        return

    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_chars = total_chars / len(documents) if documents else 0

    print(f"\n📊 {source_type.title()} Documents Summary:")
    print(f"  • Total documents: {len(documents)}")
    print(f"  • Total characters: {total_chars:,}")
    print(f"  • Average document length: {avg_chars:.0f} characters")

    # Count by source type
    source_types = {}
    for doc in documents:
        doc_type = doc.metadata.get('source_type', 'unknown')
        source_types[doc_type] = source_types.get(doc_type, 0) + 1

    if len(source_types) > 1:
        print("  • By type:")
        for doc_type, count in sorted(source_types.items()):
            print(f"    - {doc_type}: {count}")
