"""
Main entry point for Agentic RAG System.

This script orchestrates the entire RAG pipeline with flexible document sources:
1. Setup and configuration
2. User query input
2b. Document source selection (online search / local upload / both)
3. ONLINE MODE: Tavily web search + filtering + summarization + approval + download
   OR LOCAL MODE: Load documents from user-specified directory
   OR BOTH: Combine online search results with local documents
4. Content extraction and loading
5. Parallel document chunking with Ray
6. Vector embedding and storage
7. Context retrieval
8. Answer generation with numeric citations
9. Display answer with sources
10. Session summary
11. Update analytics
12. Download directory cleanup confirmation (online sources only)
"""

import uuid
import shutil
from datetime import datetime
from openai import OpenAI

from config import get_config
from src.utils import (
    setup_logging,
    get_logger,
    CostTracker,
    SessionData,
    update_analytics,
    display_progress,
    prompt_source_approval,
    print_answer,
    print_session_summary,
    print_error,
    print_success,
    print_header,
    log_session_start,
    log_session_end
)
from src.ingestion import (
    TavilySearchClient,
    filter_sources_by_relevance,
    deduplicate_sources,
    summarize_sources_sync,
    download_articles_from_sources,
    load_pdf_source,
    load_pdf_from_file,
    load_text_file,
    load_article,
    load_youtube_video,
    get_document_source_mode,
    load_local_documents,
    print_document_summary
)
from src.vectorstore import (
    OpenAIEmbedding,
    ChromaVectorStore,
    parallel_chunk_documents,
    shutdown_ray
)
from src.generation import RAGAnswerGenerator

logger = get_logger(__name__)


def main():
    """Main RAG pipeline execution."""
    session_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()

    try:
        # Setup logging first
        setup_logging()

        # Print header
        print_header("AGENTIC RAG SYSTEM v1.0")

        # 1. Setup and configuration
        display_progress("Initializing system...")
        config = get_config()

        # Validate API keys
        is_valid, missing_keys = config.validate_api_keys()
        if not is_valid:
            print_error(
                "Missing required API keys",
                f"Please set the following in .env file: {', '.join(missing_keys)}"
            )
            return

        # Initialize clients
        cost_tracker = CostTracker()
        openai_client = OpenAI(api_key=config.api.openai_api_key)
        search_client = TavilySearchClient(
            api_key=config.api.tavily_api_key,
            cost_tracker=cost_tracker
        )

        # Initialize embedding model
        display_progress("Loading embedding model...")
        embedding_model = OpenAIEmbedding(
            api_key=config.api.openai_api_key,
            model="text-embedding-3-small"
        )

        # Initialize vector store
        vector_store = ChromaVectorStore(
            persist_directory=str(config.vectorstore.persist_directory),
            collection_name=config.vectorstore.collection_name,
            embedding_model=embedding_model
        )

        print_success("System initialized successfully")

        # 2. Get user query
        print("\n")
        query = input("What would you like to research? ").strip()

        if not query:
            print_error("No query provided")
            return

        log_session_start(session_id, query)

        # 2b. Get document source mode
        source_mode = get_document_source_mode()

        # Initialize variables for tracking
        search_results = []
        approved_sources = []
        query_download_dir = None
        documents = []

        # 3. Process online sources (if online or both)
        if source_mode in ['online', 'both']:
            # 3a. Search for sources
            display_progress(f"Searching for relevant sources...")
            search_results = search_client.search(
                query=query,
                max_results=100,  # Get up to 100 sources
                search_depth="advanced"
            )

            if not search_results:
                print_error("No sources found for your query")
                if source_mode == 'online':
                    return
                # Continue to local documents if mode is 'both'
            else:
                print_success(f"Found {len(search_results)} sources")

                # 4. Filter sources by relevance
                display_progress("Filtering sources by relevance...")
                filtered_sources = filter_sources_by_relevance(
                    query=query,
                    sources=search_results,
                    openai_client=openai_client,
                    threshold=config.search.relevance_threshold
                )
                filtered_sources = deduplicate_sources(filtered_sources)

                if not filtered_sources:
                    print_error("No relevant sources found after filtering")
                    if source_mode == 'online':
                        return
                else:
                    print_success(f"Filtered to {len(filtered_sources)} relevant sources")

                    # 5. Generate summaries
                    display_progress("Generating source summaries...")
                    summaries = summarize_sources_sync(
                        sources=filtered_sources,
                        query=query,
                        client=openai_client,
                        cost_tracker=cost_tracker,
                        model=config.model.summarization_model
                    )

                    # 6. Get user approval
                    approved_sources = prompt_source_approval(summaries)

                    if not approved_sources:
                        print_error("No sources approved.")
                        if source_mode == 'online':
                            return
                    else:
                        print_success(f"Processing {len(approved_sources)} approved sources")

                        # 7. Download/parse articles - ensure minimum 10 successful saves
                        display_progress("Saving articles (PDFs: download, Articles: parse & save as .txt)...")
                        print("  Target: 10 successfully saved articles")
                        saved_articles = download_articles_from_sources(
                            sources=approved_sources,
                            query=query,
                            base_dir="./data",
                            min_downloads=10
                        )

                        # Track download directory for cleanup later
                        from src.ingestion.article_downloader import create_query_directory
                        query_download_dir = create_query_directory("./data", query)

                        # Create mapping of URLs to local file paths
                        downloaded_map = {source.url: file_path for source, file_path in saved_articles}

                        # Count PDFs vs TXT files
                        pdf_count = sum(1 for _, path in saved_articles if path.suffix == '.pdf')
                        txt_count = sum(1 for _, path in saved_articles if path.suffix == '.txt')

                        if len(saved_articles) >= 10:
                            print_success(f"Successfully saved {len(saved_articles)} articles (PDFs: {pdf_count}, Parsed: {txt_count})")
                        elif len(saved_articles) > 0:
                            print(f"⚠️  Only saved {len(saved_articles)} articles (target: 10) - PDFs: {pdf_count}, Parsed: {txt_count}")
                            proceed = input("Proceed with fewer articles? (y/n): ").strip().lower()
                            if proceed != 'y':
                                print_error("Insufficient articles saved.")
                                if source_mode == 'online':
                                    return
                        else:
                            print_error("No articles saved.")
                            if source_mode == 'online':
                                return

                        # 8. Load content from saved sources (PDFs and parsed articles)
                        display_progress("Loading content from saved sources...")

                        for idx, source in enumerate(approved_sources, 1):
                            try:
                                display_progress(f"Loading source {idx}/{len(approved_sources)}: {source.source_type}", idx, len(approved_sources))

                                # Check if we have this source saved locally
                                if source.url in downloaded_map:
                                    local_path = downloaded_map[source.url]
                                    logger.info(f"Loading from saved file: {local_path}")

                                    # Determine file type and load accordingly
                                    if local_path.suffix == '.pdf':
                                        doc = load_pdf_from_file(
                                            str(local_path),
                                            source_url=source.url
                                        )
                                    elif local_path.suffix == '.txt':
                                        doc = load_text_file(
                                            str(local_path),
                                            source_url=source.url
                                        )
                                    else:
                                        raise ValueError(f"Unsupported file type: {local_path.suffix}")

                                # Load from URL if not saved locally
                                elif source.source_type == "pdf":
                                    doc = load_pdf_source(source.url)
                                elif source.source_type == "video":
                                    doc = load_youtube_video(source.url)
                                else:  # article
                                    doc = load_article(source.url)

                                documents.append(doc)
                                print_success(f"  Loaded: {source.title[:60]}...")

                            except Exception as e:
                                logger.warning(f"Failed to load source {source.url}: {str(e)}")
                                print(f"  ⚠️  Skipped: {source.title[:60]}... (error: {str(e)[:50]})")
                                continue

                        if documents:
                            print_success(f"Successfully loaded {len(documents)} online sources")
                            print_document_summary(documents, "online")

        # 3b. Load local documents (if local or both)
        if source_mode in ['local', 'both']:
            local_documents = load_local_documents()

            if local_documents:
                documents.extend(local_documents)
                print_document_summary(local_documents, "local")
            elif source_mode == 'local':
                print_error("No local documents loaded. Cannot proceed.")
                return

        # Verify we have documents to process
        if not documents:
            print_error("No documents available for processing")
            return

        # Show combined summary if both modes
        if source_mode == 'both' and documents:
            print_document_summary(documents, "combined")

        # 9. Chunk documents in parallel using Ray
        display_progress("Chunking documents in parallel...")
        chunks = parallel_chunk_documents(
            documents=documents,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            num_workers=4
        )

        print_success(f"Created {len(chunks)} chunks")

        # 10. Add to vector store
        display_progress("Creating embeddings and storing in vector database...")
        vector_store.add_documents(chunks)

        stats = vector_store.get_collection_stats()
        print_success(
            f"Vector store updated: {stats['total_documents']} total documents"
        )

        # 11. Retrieve relevant context
        display_progress("Retrieving relevant context for your query...")
        context_docs_with_scores = vector_store.similarity_search(
            query=query,
            k=config.retrieval.retrieval_k
        )

        context_docs = [doc for doc, _ in context_docs_with_scores]

        if not context_docs:
            print_error("No relevant context found")
            return

        print_success(f"Retrieved {len(context_docs)} relevant chunks")

        # 12. Generate answer
        display_progress("Generating answer...")
        answer_generator = RAGAnswerGenerator(
            client=openai_client,
            model=config.model.generation_model,
            cost_tracker=cost_tracker
        )

        answer = answer_generator.generate_answer(
            query=query,
            context_documents=context_docs,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens
        )

        # 13. Display answer
        print_answer(answer)

        # 14. Session summary
        duration = (datetime.now() - start_time).total_seconds()
        session_costs = cost_tracker.get_session_costs()

        print_session_summary(
            sources_found=len(search_results) if search_results else 0,
            sources_approved=len(approved_sources) if approved_sources else 0,
            sources_processed=len(documents),
            chunks_created=len(chunks),
            total_cost=session_costs['total'],
            duration=duration
        )

        # 15. Update analytics
        session_data = SessionData(
            session_id=session_id,
            query=query,
            timestamp=start_time.isoformat(),
            sources_found=len(search_results) if search_results else 0,
            sources_approved=len(approved_sources) if approved_sources else 0,
            sources_processed=len(documents),
            chunks_created=len(chunks),
            answer_length=len(answer.answer),
            total_cost=session_costs['total'],
            models_used=session_costs['by_model'],
            duration=duration,
            success=True
        )

        update_analytics(session_data)
        log_session_end(session_id, duration, session_costs['total'], len(documents))

        print_success("\nSession completed successfully!")

        # 16. Cleanup confirmation (only for online sources)
        if query_download_dir and query_download_dir.exists() and any(query_download_dir.iterdir()):
            print(f"\n📁 Downloaded articles location: {query_download_dir}")
            cleanup = input("Delete downloaded articles directory? (y/n): ").strip().lower()
            if cleanup == 'y':
                try:
                    shutil.rmtree(query_download_dir)
                    print_success(f"Cleaned up download directory: {query_download_dir}")
                    logger.info(f"Cleaned up download directory: {query_download_dir}")
                except Exception as e:
                    print(f"⚠️  Failed to delete directory: {e}")
                    logger.warning(f"Failed to delete directory: {e}")
            else:
                print(f"📁 Articles preserved at: {query_download_dir}")

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        logger.info("Session interrupted by user")

    except Exception as e:
        logger.error(f"Session failed: {str(e)}", exc_info=True)
        print_error(f"An error occurred: {str(e)}")

    finally:
        # Cleanup Ray resources
        shutdown_ray()


if __name__ == "__main__":
    main()
