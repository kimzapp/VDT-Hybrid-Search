# ==============================================================================
# VDT Hybrid Search — FastAPI Backend
# ==============================================================================

import os
import sqlite3
import sys
import time
import traceback
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Ensure that the scripts directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_path = os.path.join(PROJECT_ROOT, "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

import config
from retriever import BM25SRetriever, DenseFaissRetriever, CrossEncoderReranker
from fusion import get_fusion_fn, list_strategies

app = FastAPI(
    title="VDT Hybrid Search API",
    description="Backend API for Sparse (BM25S) + Dense (FAISS) Hybrid Search Engine",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded objects
bm25_retriever = None
dense_retriever = None
reranker = None
docs_store = None
metadata_db_path = None
dense_loaded = False
dense_load_error = None

# Lifespan/Startup Initialization
@app.on_event("startup")
def startup_event():
    global bm25_retriever, dense_retriever, reranker, docs_store, metadata_db_path, dense_loaded, dense_load_error

    # 1. Load Sparse Retriever (BM25S)
    print("Loading sparse index (BM25S)...")
    if os.path.exists(config.SPARSE_INDEX_DIR):
        try:
            bm25_retriever = BM25SRetriever.load(
                config.SPARSE_INDEX_DIR,
                mmap=True,
                tokenize_kwargs={},
            )
            print("Successfully loaded sparse index.")
        except Exception as e:
            print(f"Error loading sparse index from {config.SPARSE_INDEX_DIR}: {e}")
            traceback.print_exc()
    else:
        print(f"Warning: Sparse index directory {config.SPARSE_INDEX_DIR} does not exist.")

    # 2. Load Dense Retriever (FAISS)
    print("Loading dense index (FAISS)...")
    if os.path.exists(config.DENSE_INDEX_DIR):
        # Try loading on configured device (e.g., GPU/CUDA)
        try:
            dense_retriever = DenseFaissRetriever.load(
                index_dir=config.DENSE_INDEX_DIR,
                model_name=config.EMBEDDING_MODEL,
                device=config.DEVICE,
                use_gpu=config.DENSE_USE_GPU,
            )
            dense_loaded = True
            print("Successfully loaded dense index on primary device.")
        except Exception as e1:
            print(f"Could not load dense retriever on {config.DEVICE}: {e1}")
            print("Attempting to load dense index on CPU as a fallback...")
            try:
                dense_retriever = DenseFaissRetriever.load(
                    index_dir=config.DENSE_INDEX_DIR,
                    model_name=config.EMBEDDING_MODEL,
                    device="cpu",
                    use_gpu=False,
                )
                dense_loaded = True
                print("Successfully loaded dense index on CPU.")
            except Exception as e2:
                dense_load_error = f"Primary load error: {e1}. Fallback load error: {e2}"
                print(f"Failed to load dense retriever entirely: {dense_load_error}")
                traceback.print_exc()
    else:
        dense_load_error = f"Dense index directory {config.DENSE_INDEX_DIR} does not exist."
        print(f"Warning: {dense_load_error}")

    # 2.5 Load Reranker
    print("Loading reranker...")
    try:
        reranker = CrossEncoderReranker(model_name=config.RERANKER_MODEL, device=config.DEVICE)
        print("Successfully loaded reranker.")
    except Exception as e:
        print(f"Error loading reranker: {e}")
        traceback.print_exc()

    # 3. Load ir_datasets docs store for passage lookups (disk-backed, low RAM)
    print(f"Loading ir_datasets docs store for corpus: {config.CORPUS_ID}...")
    try:
        import ir_datasets
        passages = ir_datasets.load(config.CORPUS_ID)
        docs_store = passages.docs_store()
        print("Successfully initialized disk-backed docs store.")
    except Exception as e:
        print(f"Error loading docs store: {e}. Passages text will not be displayed.")
        traceback.print_exc()

    # 4. Load metadata SQLite database path
    if os.path.exists(config.METADATA_DB):
        metadata_db_path = config.METADATA_DB
        print(f"Metadata SQLite database found: {metadata_db_path}")
    else:
        print(f"Warning: Metadata database {config.METADATA_DB} not found. Metadata will not be displayed.")


# Request/Response schemas
class SearchRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", description="sparse | dense | hybrid")
    top_k: int = Field(default=10, ge=1, le=100)
    fusion_strategy: str = Field(default="rrf", description="rrf | weighted_sum | combsum | combmnz | borda | union")
    rrf_k: int = Field(default=60, ge=1)
    fusion_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank: bool = Field(default=False, description="Enable CrossEncoder reranking")
    rerank_top_k: int = Field(default=100, ge=1, le=1000)
    wcr: bool = Field(default=False, description="Enable Weighted Combination of two-stage Ranking")
    wcr_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for retrieval score in WCR")
    date_from: Optional[str] = Field(default=None, description="Filter: min written_date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(default=None, description="Filter: max written_date (YYYY-MM-DD)")

class SearchResultItem(BaseModel):
    rank: int
    doc_id: str
    score: float
    text: str
    author_name: Optional[str] = None
    written_date: Optional[str] = None

class LatencyStats(BaseModel):
    total_ms: float
    sparse_ms: Optional[float] = None
    dense_ms: Optional[float] = None
    fusion_ms: Optional[float] = None
    rerank_ms: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    mode: str
    top_k: int
    results: List[SearchResultItem]
    latency: LatencyStats
    fusion_strategy: Optional[str] = None
    num_results: int
    total_before_filter: int = 0

class AuthorPassageItem(BaseModel):
    doc_id: str
    text: str
    author_name: str
    written_date: Optional[str] = None

class AuthorPassagesResponse(BaseModel):
    author_name: str
    passages: List[AuthorPassageItem]
    page: int
    page_size: int
    total: int
    total_pages: int


# Endpoints
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "sparse_index_loaded": bm25_retriever is not None,
        "dense_index_loaded": dense_loaded,
        "docs_store_loaded": docs_store is not None,
    }

@app.get("/api/config")
def get_backend_config():
    return {
        "corpus_id": config.CORPUS_ID,
        "sparse_index_dir": config.SPARSE_INDEX_DIR,
        "dense_index_dir": config.DENSE_INDEX_DIR,
        "embedding_model": config.EMBEDDING_MODEL,
        "device": config.DEVICE if dense_loaded else "N/A",
        "dense_loaded": dense_loaded,
        "dense_load_error": dense_load_error,
        "available_fusion_strategies": list_strategies(),
    }

def _batch_lookup_metadata(doc_ids: list) -> dict:
    """
    Batch-lookup author_name and written_date from the SQLite metadata DB.
    Returns: {doc_id: {"author_name": str, "written_date": str}, ...}
    """
    if not metadata_db_path or not doc_ids:
        return {}

    result = {}
    try:
        conn = sqlite3.connect(metadata_db_path, timeout=5)
        conn.execute("PRAGMA query_only=ON;")
        cursor = conn.cursor()

        # Use batched IN clause queries (SQLite variable limit is ~999)
        batch_size = 900
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))
            cursor.execute(
                f"SELECT doc_id, author_name, written_date FROM passage_metadata_full WHERE doc_id IN ({placeholders})",
                batch,
            )
            for row in cursor.fetchall():
                result[row[0]] = {
                    "author_name": row[1],
                    "written_date": row[2],
                }

        conn.close()
    except Exception as e:
        print(f"Warning: metadata lookup failed: {e}")

    return result


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t_start = time.perf_counter()

    mode = req.mode.lower().strip()
    if mode not in ("sparse", "dense", "hybrid"):
        raise HTTPException(status_code=400, detail="Invalid search mode. Choose from: 'sparse', 'dense', 'hybrid'")

    if mode == "sparse" and bm25_retriever is None:
        raise HTTPException(status_code=503, detail="Sparse index is not loaded/available on backend.")
    if mode == "dense" and not dense_loaded:
        raise HTTPException(status_code=503, detail=f"Dense index is not available. Error: {dense_load_error}")
    if mode == "hybrid" and (bm25_retriever is None or not dense_loaded):
        available = []
        if bm25_retriever is not None: available.append("sparse")
        if dense_loaded: available.append("dense")
        raise HTTPException(
            status_code=503,
            detail=f"Hybrid search requires both sparse and dense indices loaded. Loaded: {available}"
        )

    sparse_ms = None
    dense_ms = None
    fusion_ms = None
    rerank_ms = None
    final_results = {}  # doc_id -> score

    # 1. Sparse Search
    if mode in ("sparse", "hybrid"):
        t0 = time.perf_counter()
        try:
            # search takes a list of queries, returns list of dicts.
            sparse_res = bm25_retriever.search([req.query], top_k=req.top_k)[0]
            final_results = sparse_res
            sparse_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error executing sparse search: {str(e)}")

    # 2. Dense Search
    if mode in ("dense", "hybrid"):
        t0 = time.perf_counter()
        try:
            dense_res = dense_retriever.search([req.query], top_k=req.top_k)[0]
            if mode == "dense":
                final_results = dense_res
            dense_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            error_msg = str(e)
            is_cuda_err = any(word in error_msg.lower() for word in ("cuda", "device", "out of memory", "alloc"))
            
            if is_cuda_err and dense_retriever is not None and (getattr(dense_retriever, "device", "") == "cuda" or getattr(dense_retriever, "use_gpu", False)):
                print("CUDA/GPU error during dense search. Dynamically falling back to CPU...")
                try:
                    # 1. Move SentenceTransformer model to CPU
                    if hasattr(dense_retriever, "model") and dense_retriever.model is not None:
                        dense_retriever.model = dense_retriever.model.to("cpu")
                    dense_retriever.device = "cpu"
                    
                    # 2. Move FAISS index to CPU
                    if hasattr(dense_retriever, "index") and dense_retriever.index is not None:
                        if getattr(dense_retriever, "_index_on_gpu", False):
                            import faiss
                            dense_retriever.index = faiss.index_gpu_to_cpu(dense_retriever.index)
                            dense_retriever._index_on_gpu = False
                    dense_retriever.use_gpu = False
                    
                    # 3. Retry search on CPU
                    t0 = time.perf_counter()
                    dense_res = dense_retriever.search([req.query], top_k=req.top_k)[0]
                    if mode == "dense":
                        final_results = dense_res
                    dense_ms = (time.perf_counter() - t0) * 1000.0
                    print("Successfully recovered and executed dense search on CPU.")
                except Exception as recovery_err:
                    print(f"Failed to fall back to CPU: {recovery_err}")
                    traceback.print_exc()
                    raise HTTPException(
                        status_code=500,
                        detail=f"Dense search failed on GPU ({error_msg}) and CPU fallback also failed: {str(recovery_err)}"
                    )
            else:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error executing dense search: {error_msg}")

    # 3. Fusion (Hybrid)
    if mode == "hybrid":
        t0 = time.perf_counter()
        try:
            fusion_fn = get_fusion_fn(req.fusion_strategy)
            fusion_kwargs = {
                "rrf_k": req.rrf_k,
                "k": req.rrf_k,
                "alpha": req.fusion_alpha,
                "fusion_alpha": req.fusion_alpha,
            }
            # fusion_fn operates on lists of query runs
            fused_res = fusion_fn(
                sparse_results=[sparse_res],
                dense_results=[dense_res],
                top_k=req.top_k,
                **fusion_kwargs
            )[0]
            final_results = fused_res
            fusion_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error executing fusion: {str(e)}")

    # 3.5 Reranking
    if req.rerank:
        if reranker is None:
            raise HTTPException(status_code=503, detail="Reranker is not loaded on backend.")
        
        t0_rerank = time.perf_counter()
        try:
            sorted_for_rerank = sorted(final_results.items(), key=lambda item: item[1], reverse=True)[:req.rerank_top_k]
            temp_corpus = {}
            for doc_id, _ in sorted_for_rerank:
                if docs_store is not None:
                    try:
                        doc = docs_store.get(doc_id)
                        if doc is not None:
                            temp_corpus[doc_id] = doc.text
                    except Exception:
                        pass
                        
            final_results = reranker.rerank(
                queries=[req.query],
                run_dict_list=[final_results],
                corpus=temp_corpus,
                top_k=req.rerank_top_k,
                wcr=req.wcr,
                wcr_alpha=req.wcr_alpha
            )[0]
            rerank_ms = (time.perf_counter() - t0_rerank) * 1000.0
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error executing reranking: {str(e)}")

    # 4. Format outputs and lookup passage texts
    results_list = []
    # Sort doc_ids by score descending
    sorted_docs = sorted(final_results.items(), key=lambda item: item[1], reverse=True)
    total_before_filter = len(sorted_docs)

    # Batch-fetch metadata from SQLite
    metadata_map = _batch_lookup_metadata([doc_id for doc_id, _ in sorted_docs])

    # Apply date filter if provided
    if req.date_from or req.date_to:
        filtered_docs = []
        for doc_id, score in sorted_docs:
            meta = metadata_map.get(doc_id, {})
            wd = meta.get("written_date", "")
            if req.date_from and wd < req.date_from:
                continue
            if req.date_to and wd > req.date_to:
                continue
            filtered_docs.append((doc_id, score))
        sorted_docs = filtered_docs

    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
        # Fetch text from docs store
        text = "Passage text database not loaded."
        if docs_store is not None:
            try:
                doc = docs_store.get(doc_id)
                if doc is not None:
                    text = doc.text
                else:
                    text = f"Document {doc_id} not found in {config.CORPUS_ID} dataset."
            except Exception as lookup_err:
                text = f"Error retrieving text: {str(lookup_err)}"

        # Get metadata if available
        meta = metadata_map.get(doc_id, {})

        results_list.append(
            SearchResultItem(
                rank=rank,
                doc_id=doc_id,
                score=float(score),
                text=text,
                author_name=meta.get("author_name"),
                written_date=meta.get("written_date"),
            )
        )

    t_total = (time.perf_counter() - t_start) * 1000.0

    return SearchResponse(
        query=req.query,
        mode=mode,
        top_k=req.top_k,
        results=results_list,
        latency=LatencyStats(
            total_ms=t_total,
            sparse_ms=sparse_ms,
            dense_ms=dense_ms,
            fusion_ms=fusion_ms,
            rerank_ms=rerank_ms,
        ),
        fusion_strategy=req.fusion_strategy if mode == "hybrid" else None,
        num_results=len(results_list),
        total_before_filter=total_before_filter,
    )


@app.get("/api/author/passages")
def get_author_passages(
    author_name: str = Query(..., description="Author name to look up"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """Return paginated passages for a given author."""
    if not metadata_db_path:
        raise HTTPException(status_code=503, detail="Metadata database is not loaded.")

    try:
        conn = sqlite3.connect(metadata_db_path, timeout=5)
        conn.execute("PRAGMA query_only=ON;")
        cursor = conn.cursor()

        # Get total count
        cursor.execute(
            "SELECT COUNT(*) FROM passage_metadata_full WHERE author_name = ?",
            (author_name,),
        )
        total = cursor.fetchone()[0]

        if total == 0:
            conn.close()
            return AuthorPassagesResponse(
                author_name=author_name,
                passages=[],
                page=page,
                page_size=page_size,
                total=0,
                total_pages=0,
            )

        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size

        cursor.execute(
            "SELECT doc_id, author_name, written_date FROM passage_metadata_full "
            "WHERE author_name = ? ORDER BY written_date DESC LIMIT ? OFFSET ?",
            (author_name, page_size, offset),
        )
        rows = cursor.fetchall()
        conn.close()

        passages = []
        for row in rows:
            doc_id, a_name, written_date = row
            text = "Passage text database not loaded."
            if docs_store is not None:
                try:
                    doc_obj = docs_store.get(str(doc_id))
                    if doc_obj is not None:
                        text = doc_obj.text
                    else:
                        text = f"Document {doc_id} not found."
                except Exception:
                    text = f"Error retrieving text for {doc_id}."

            passages.append(AuthorPassageItem(
                doc_id=str(doc_id),
                text=text,
                author_name=a_name,
                written_date=written_date,
            ))

        return AuthorPassagesResponse(
            author_name=author_name,
            passages=passages,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        )

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# Serve frontend static files
frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory {frontend_dir} not found. Static files will not be served.")


if __name__ == "__main__":
    print(f"Starting server on http://{config.HOST}:{config.PORT}...")
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=True)
