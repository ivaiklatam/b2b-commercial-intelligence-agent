"""
RAG tool using ChromaDB and Google Gemini embeddings.
Indexes portfolio documents and retrieves relevant context
for recommendation generation.
"""

import json
import os
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool


CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "b2b_portfolio"
_vectorstore = None


def _load_portfolio() -> list:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "portfolio.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["documents"]


def _get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
    )


def _compute_age_days(timestamp_str: str) -> int:
    """Computes how many days old a document is."""
    try:
        doc_date = datetime.strptime(timestamp_str, "%Y-%m-%d")
        return (datetime.now() - doc_date).days
    except Exception:
        return 0


def initialize_rag() -> None:
    """
    Initializes ChromaDB vectorstore with portfolio documents.
    Chunks documents at section level and adds metadata:
    timestamp, doc_type, sector, age_days.
    """
    global _vectorstore

    embeddings = _get_embeddings()

    # Check if already initialized
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("   📚 Loading existing ChromaDB index...")
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        return

    print("   📚 Building ChromaDB index from portfolio...")
    documents = _load_portfolio()
    langchain_docs = []

    for doc in documents:
        age_days = _compute_age_days(doc.get("timestamp", "2026-01-01"))
        sectors = doc.get("sector", [])
        if isinstance(sectors, list):
            sectors_str = ",".join(sectors)
        else:
            sectors_str = str(sectors)

        # Main content chunk
        main_content = (
            f"Title: {doc['title']}\n"
            f"Type: {doc['doc_type']}\n"
            f"Sectors: {sectors_str}\n"
            f"Content: {doc['content']}"
        )
        langchain_docs.append(Document(
            page_content=main_content,
            metadata={
                "doc_id": doc["id"],
                "doc_type": doc["doc_type"],
                "title": doc["title"],
                "sectors": sectors_str,
                "timestamp": doc.get("timestamp", ""),
                "age_days": age_days,
                "partner": doc.get("partner", ""),
                "chunk_type": "main"
            }
        ))

        # Outcomes chunk (separate for better retrieval)
        if doc.get("outcomes"):
            outcomes_content = (
                f"Title: {doc['title']}\n"
                f"Type: {doc['doc_type']}\n"
                f"Measured outcomes and results: {doc['outcomes']}"
            )
            langchain_docs.append(Document(
                page_content=outcomes_content,
                metadata={
                    "doc_id": doc["id"],
                    "doc_type": doc["doc_type"],
                    "title": doc["title"],
                    "sectors": sectors_str,
                    "timestamp": doc.get("timestamp", ""),
                    "age_days": age_days,
                    "partner": doc.get("partner", ""),
                    "chunk_type": "outcomes"
                }
            ))

    _vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print(f"   ✅ ChromaDB indexed {len(langchain_docs)} chunks from "
          f"{len(documents)} documents")


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        initialize_rag()
    return _vectorstore


@tool
def retrieve_portfolio_context(
    query: str,
    sector: str = "",
    top_k: int = 5
) -> str:
    """
    Retrieves relevant portfolio context for a commercial recommendation.
    Query should combine signal type, sector and opportunity category.
    Returns top matching documents with metadata including age warnings.
    """
    try:
        vs = _get_vectorstore()
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "config.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        pricing_max_age = config["rag"]["pricing_content_max_age_days"]
        case_max_age = config["rag"]["case_study_max_age_days"]

        # Retrieve documents
        results = vs.similarity_search_with_score(query, k=top_k)

        retrieved = []
        for doc, score in results:
            meta = doc.metadata
            age_days = meta.get("age_days", 0)
            doc_type = meta.get("doc_type", "")

            # Timestamp warnings
            warnings = []
            if doc_type == "product" and age_days > pricing_max_age:
                warnings.append(
                    f"⚠️ Pricing content may be outdated ({age_days} days old)"
                )
            if doc_type == "case_study" and age_days > case_max_age:
                warnings.append(
                    f"⚠️ Case study is older than 18 months ({age_days} days old)"
                )

            retrieved.append({
                "title": meta.get("title", ""),
                "doc_type": doc_type,
                "sectors": meta.get("sectors", ""),
                "partner": meta.get("partner", ""),
                "timestamp": meta.get("timestamp", ""),
                "age_days": age_days,
                "relevance_score": round(1 - score, 3),
                "content": doc.page_content[:400],
                "warnings": warnings
            })

        has_evidence = len(retrieved) > 0
        strong_evidence = [r for r in retrieved if r["relevance_score"] >= 0.5]

        return json.dumps({
            "query": query,
            "sector": sector,
            "has_evidence": has_evidence,
            "strong_evidence_count": len(strong_evidence),
            "results": retrieved
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "query": query,
            "has_evidence": False,
            "error": str(e)
        })