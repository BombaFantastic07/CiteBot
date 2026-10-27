"""Model factory: configures LlamaIndex global LLM + embedding model for the chosen provider."""
from __future__ import annotations

from llama_index.core import Settings

from .config import AppConfig


def build_llm(cfg: AppConfig):
    if cfg.provider == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(model=cfg.openai_llm_model, temperature=0.0)

    from llama_index.llms.bedrock_converse import BedrockConverse

    return BedrockConverse(
        model=cfg.bedrock_llm_model, region_name=cfg.aws_region, temperature=0.0
    )


def build_embed_model(cfg: AppConfig):
    if cfg.provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(
            model=cfg.openai_embed_model, embed_batch_size=cfg.embed_batch_size
        )

    from llama_index.embeddings.bedrock import BedrockEmbedding

    return BedrockEmbedding(
        model_name=cfg.bedrock_embed_model,
        region_name=cfg.aws_region,
        embed_batch_size=min(cfg.embed_batch_size, 32),
    )


def configure_models(cfg: AppConfig) -> None:
    """Set LlamaIndex globals. Temperature 0 keeps answers deterministic and grounded."""
    Settings.llm = build_llm(cfg)
    Settings.embed_model = build_embed_model(cfg)
