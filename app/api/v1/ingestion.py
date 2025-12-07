from typing import Dict, Optional
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from app.services.text_extractor import extract_text_from_file
from app.utils.chunking import chunk_text_fixed, chunk_text_sentences, chunk_text_recursive
from app.services.embeddings import EmbeddingService
from app.services.vectorstore import VectorStore
from app.utils.db import get_db_session, init_db, FileChunkMeta, Base

router = APIRouter()

# Ensure DB created on import (simple local startup init)
init_db()

@router.post("", response_model=Dict)
async def ingest_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Query("fixed", regex="^(fixed|sentence|recursive)$"),
    chunk_size: int = Query(500, gt=0),
    chunk_overlap: int = Query(50, ge=0),
) -> Dict:
    """
    Ingest a PDF or TXT file, extract text, chunk, embed, and store vectors + metadata.

    - chunking_strategy: "fixed" (fixed token-size approx), "sentence" (sentence-based), or "recursive" (recursive character splitter with overlap)
    - chunk_size: tokens for chunking strategies (approx by whitespace tokens)
    - chunk_overlap: number of tokens to overlap between chunks (only used for "recursive" strategy)
    """
    if file.content_type not in ("application/pdf", "text/plain"):
        raise HTTPException(status_code=400, detail="Only .pdf or .txt files supported")

    raw_text = await extract_text_from_file(file)
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from file")

    # Chunking
    if chunking_strategy == "fixed":
        chunks = chunk_text_fixed(raw_text, chunk_size=chunk_size)
    elif chunking_strategy == "sentence":
        chunks = chunk_text_sentences(raw_text, chunk_size=chunk_size)
    elif chunking_strategy == "recursive":
        chunks = chunk_text_recursive(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown chunking strategy: {chunking_strategy}")

    # Embeddings + store vectors
    emb_service = EmbeddingService()
    vs = VectorStore()
    saved_meta = []
    # store each chunk: generate embedding and upsert to vector DB, save metadata in SQL
    for idx, chunk_text in enumerate(chunks):
        emb = emb_service.embed_text(chunk_text)
        vec_id = vs.upsert_vector(vector=emb, payload={"file_name": file.filename, "chunk_id": idx, "text": chunk_text})
        # Save metadata to SQL DB
        session = get_db_session()
        meta = FileChunkMeta(file_name=file.filename, chunk_id=idx, chunk_text=chunk_text, embedding_id=str(vec_id))
        session.add(meta)
        session.commit()
        saved_meta.append({"chunk_id": idx, "embedding_id": str(vec_id)})

    return JSONResponse({"status": "success", "file": file.filename, "chunks": len(chunks), "saved": saved_meta})