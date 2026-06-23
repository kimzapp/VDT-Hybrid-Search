from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from sentence_transformers import CrossEncoder

@dataclass(frozen=True)
class RerankerModelConfig:
    model_id: str
    trust_remote_code: bool = False
    max_length: Optional[int] = 512
    default_batch_size: int = 32
    note: str = ""

RERANKER_MODEL_REGISTRY: Dict[str, RerankerModelConfig] = {
    "bge_reranker_base": RerankerModelConfig(
        model_id="BAAI/bge-reranker-base",
        note="Base BGE Reranker. Good balance of speed and accuracy."
    ),
    "bge_reranker_large": RerankerModelConfig(
        model_id="BAAI/bge-reranker-large",
        note="Large BGE Reranker. High accuracy."
    ),
    "bge_reranker_v2_m3": RerankerModelConfig(
        model_id="BAAI/bge-reranker-v2-m3",
        note="BGE Reranker V2 M3 (Multilingual)."
    ),
}

def create_reranker_model(
    model_id: str,
    device: str = "cuda",
    trust_remote_code: bool = False,
    model_kwargs: dict = None,
):
    """Create a reranker model from registry key or HuggingFace model ID.

    Returns:
        (model, config): CrossEncoder model and its RerankerModelConfig.
    """
    model_kwargs = model_kwargs or {}

    # --- Resolve config ---
    if model_id in RERANKER_MODEL_REGISTRY:
        config = RERANKER_MODEL_REGISTRY[model_id]
        model_name = config.model_id
        trust_remote_code = config.trust_remote_code
    else:
        # Treat model_id as a raw HuggingFace model name / local path
        model_name = model_id
        config = RerankerModelConfig(
            model_id=model_id,
            trust_remote_code=trust_remote_code,
        )

    # --- Build CrossEncoder ---
    model = CrossEncoder(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        max_length=config.max_length,
        model_kwargs=model_kwargs,
    )

    return model, config
