# -----------------------------------------------------------------------------
#  Model registry
# -----------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer, models
    


@dataclass(frozen=True)
class EmbeddingModelConfig:
    model_id: str
    doc_prefix: str = ""
    query_prefix: str = ""
    doc_prompt_name: Optional[str] = None
    query_prompt_name: Optional[str] = None
    pool_type: Optional[str] = None
    trust_remote_code: bool = False
    normalize: bool = True
    max_seq_length: Optional[int] = None
    # Extra kwargs for model.encode(...). If non-empty, we avoid encode_multi_process
    # because not every SentenceTransformers version forwards custom kwargs there.
    doc_encode_kwargs: Dict[str, Any] = field(default_factory=dict)
    query_encode_kwargs: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


MODEL_REGISTRY: Dict[str, EmbeddingModelConfig] = {
    # Qwen3 Embedding: strong upper-bound candidates. Documents are raw text;
    # queries should use prompt_name="query" when supported by SentenceTransformers.
    "qwen3_8b": EmbeddingModelConfig(
        model_id="Qwen/Qwen3-Embedding-8B",
        query_prompt_name="query",
        trust_remote_code=True,
        normalize=True,
        note="Very strong but heavy. Use as upper-bound if hardware allows.",
    ),
    "qwen3_4b": EmbeddingModelConfig(
        model_id="Qwen/Qwen3-Embedding-4B",
        query_prompt_name="query",
        trust_remote_code=True,
        normalize=True,
        note="Strong Qwen3 embedding model, still heavy for full MS MARCO.",
    ),
    "qwen3_0_6b": EmbeddingModelConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        query_prompt_name="query",
        trust_remote_code=True,
        normalize=True,
        note="Recommended first high-quality Qwen3 run.",
    ),

    # Jina v3: highest-quality usage uses task adapters. This script uses the
    # direct Jina model with task-specific encode kwargs, so it may fall back to
    # single-process encoding to preserve correctness.
    "jina_v3": EmbeddingModelConfig(
        model_id="jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        normalize=True,
        max_seq_length=8192,
        doc_encode_kwargs={"task": "retrieval.passage"},
        query_encode_kwargs={"task": "retrieval.query"},
        note="Uses Jina retrieval task adapters; may avoid multi-process for correctness.",
    ),

    # Mixedbread: prompt only for query; documents use raw passage text.
    "mxbai_large": EmbeddingModelConfig(
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        query_prefix="Represent this sentence for searching relevant passages: ",
        normalize=True,
        note="Strong English retrieval model. Query prompt is important.",
    ),

    # GTE v1.5 English models.
    "gte_large": EmbeddingModelConfig(
        model_id="Alibaba-NLP/gte-large-en-v1.5",
        trust_remote_code=True,
        normalize=True,
        max_seq_length=8192,
        note="Strong long-context English baseline.",
    ),
    "gte_base": EmbeddingModelConfig(
        model_id="Alibaba-NLP/gte-base-en-v1.5",
        trust_remote_code=True,
        normalize=True,
        max_seq_length=8192,
        note="Good base-size long-context English baseline.",
    ),

    # BGE v1.5 English models. Query instruction usually helps retrieval.
    "bge_large": EmbeddingModelConfig(
        model_id="BAAI/bge-large-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        normalize=True,
        note="Strong BGE English baseline.",
    ),
    "bge_base": EmbeddingModelConfig(
        model_id="BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        normalize=True,
        note="Mid-size BGE English baseline.",
    ),
    "bge_small": EmbeddingModelConfig(
        model_id="BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        normalize=True,
        note="Your current baseline model.",
    ),

    # E5 v2: prefix is required for asymmetric retrieval.
    "e5_large": EmbeddingModelConfig(
        model_id="intfloat/e5-large-v2",
        doc_prefix="passage: ",
        query_prefix="query: ",
        normalize=True,
        note="Strong E5 baseline. Prefixes are required.",
    ),
    "e5_base": EmbeddingModelConfig(
        model_id="intfloat/e5-base-v2",
        doc_prefix="passage: ",
        query_prefix="query: ",
        normalize=True,
        note="Base-size E5 baseline. Prefixes are required.",
    ),
    "e5_small": EmbeddingModelConfig(
        model_id="intfloat/e5-small-v2",
        doc_prefix="passage: ",
        query_prefix="query: ",
        normalize=True,
        note="Small E5 baseline. Prefixes are required.",
    ),

    # Nomic: prefixes are required.
    "nomic_v15": EmbeddingModelConfig(
        model_id="nomic-ai/nomic-embed-text-v1.5",
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
        trust_remote_code=True,
        normalize=True,
        max_seq_length=8192,
        note="Requires Nomic search_query/search_document prefixes.",
    ),

    # NV-Embed-v2: High performance 7B model from NVIDIA
    "nv_embed_v2": EmbeddingModelConfig(
        model_id="nvidia/NV-Embed-v2",
        query_prefix="Instruct: Given a question, retrieve passages that answer the question.\nQuery: ",
        trust_remote_code=True,
        normalize=True,
        max_seq_length=4096, # Can support longer context up to 32k depending on VRAM
        note="State-of-the-art 7B model. Ensure you use --dtype bfloat16 or float16 and have sufficient VRAM.",
    ),

    # Linq-Embed-Mistral: Mistral based embedding model
    "linq_embed_mistral": EmbeddingModelConfig(
        model_id="Linq-AI-Research/Linq-Embed-Mistral",
        query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        trust_remote_code=True,
        normalize=True,
        max_seq_length=4096,
        note="Mistral-based 7B model. Requires bfloat16/float16.",
    ),

    # SFR-Embedding-Mistral: State-of-the-art retrieval model from Salesforce
    "sfr_embed_mistral": EmbeddingModelConfig(
        model_id="Salesforce/SFR-Embedding-Mistral",
        query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        trust_remote_code=True,
        normalize=True,
        max_seq_length=4096,
        note="Salesforce 7B Mistral-based model. Requires bfloat16/float16.",
    ),

    # E5-Mistral-7b-instruct: Mistral-based E5 model
    "e5_mistral_7b": EmbeddingModelConfig(
        model_id="intfloat/e5-mistral-7b-instruct",
        query_prompt_name="web_search_query", # Uses SentenceTransformers built-in prompt mapping
        trust_remote_code=True,
        normalize=True,
        max_seq_length=4096,
        note="E5 7B Mistral-based model. Requires bfloat16/float16.",
    ),

    # BAAI/bge-en-icl: In-Context Learning embedding model
    "bge_en_icl": EmbeddingModelConfig(
        model_id="BAAI/bge-en-icl",
        query_prefix="<instruct>Given a web search query, retrieve relevant passages that answer the query.\n<query>",
        trust_remote_code=True,
        normalize=True,
        max_seq_length=4096,
        note="BGE ICL model. Though best with FlagICLModel and examples, this uses zero-shot prefix for SentenceTransformers compatibility.",
    ),

    # Lightweight lower bound.
    "minilm_l6": EmbeddingModelConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
        note="Fast lower-bound baseline.",
    ),

    "co-condenser-marco": EmbeddingModelConfig(
        model_id="/home/rmits/.cache/modelscope/hub/models/iic/nlp_corom_sentence-embedding_english-base",
        normalize=True,
    ),

    "embedding_vi": EmbeddingModelConfig(
        model_id="bkai-foundation-models/vietnamese-bi-encoder",
        trust_remote_code=True,
        normalize=True,
    )
}

def create_embedding_model(
    model_id: str,
    pool_type: str = None,
    normalize: bool = True,
    trust_remote_code: bool = False,
    device: str = "cuda",
    model_kwargs: dict = None,
):
    """Create an embedding model from registry key or HuggingFace model ID.

    Returns:
        (model, config): SentenceTransformer model and its EmbeddingModelConfig.
        If model_id is not in the registry, config will have default values.
    """
    model_kwargs = model_kwargs or {}

    # --- Resolve config ---
    if model_id in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model_id]
        model_name = config.model_id
        normalize = config.normalize
        trust_remote_code = config.trust_remote_code
        pool_type = pool_type or config.pool_type
    else:
        # Treat model_id as a raw HuggingFace model name / local path
        model_name = model_id
        config = EmbeddingModelConfig(
            model_id=model_id,
            normalize=normalize,
            trust_remote_code=trust_remote_code,
            pool_type=pool_type,
        )

    # --- Build SentenceTransformer ---
    if pool_type is None:
        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
        )
    else:
        word_embedding_model = models.Transformer(
            model_name,
            model_kwargs=model_kwargs,
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode=pool_type,
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device=device,
        )

    # --- Apply max_seq_length from config ---
    if config.max_seq_length is not None:
        model.max_seq_length = config.max_seq_length

    return model, config
