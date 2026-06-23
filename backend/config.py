# ==============================================================================
# VDT Hybrid Search — Backend Configuration
# ==============================================================================

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Index paths (default to pre-built paths in the project root)
SPARSE_INDEX_DIR = os.environ.get(
    "SPARSE_INDEX_DIR", os.path.join(PROJECT_ROOT, "bm25_index")
)
DENSE_INDEX_DIR = os.environ.get(
    "DENSE_INDEX_DIR", os.path.join(PROJECT_ROOT, "bge_small_en_v1.5_embedding_faiss")
)

# corpus_id used to load passage text from ir_datasets
CORPUS_ID = os.environ.get("CORPUS_ID", "msmarco-passage")

# SQLite metadata database (author, written_date, etc.)
METADATA_DB = os.environ.get(
    "METADATA_DB", os.path.join(PROJECT_ROOT, "metadata_sqlite")
)

# Embedding model settings
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# DEVICE = os.environ.get("DEVICE", "cuda")  # "cuda" or "cpu"
DEVICE = 'cuda:1'
DENSE_USE_GPU = os.environ.get("DENSE_USE_GPU", "true").lower() in ("true", "1", "yes")

# Server settings
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8888))

# CORS Allowed Origins
CORS_ORIGINS = ["*"]
