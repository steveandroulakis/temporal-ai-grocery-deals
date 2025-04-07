import os
import logging
from typing import Optional

# Default embedding dimension if not set in environment
DEFAULT_OPENAI_EMBEDDING_DIMENSION = 768  # e.g., for text-embedding-3-small

def get_env_var(var_name: str, logger: logging.Logger) -> str:
    """Fetches an environment variable or raises an error if not found."""
    value = os.environ.get(var_name)
    if not value:
        logger.error(f"{var_name} environment variable not set.")
        raise ValueError(f"{var_name} environment variable must be set.")
    return value

def get_embedding_dimension(logger: logging.Logger) -> int:
    """Determines the embedding dimension from env vars or uses default."""
    dimension_str: Optional[str] = os.environ.get("OPENAI_EMBEDDING_DIMENSION")
    if not dimension_str:
        logger.info(f"OPENAI_EMBEDDING_DIMENSION not set. Using default: {DEFAULT_OPENAI_EMBEDDING_DIMENSION}")
        return DEFAULT_OPENAI_EMBEDDING_DIMENSION
    try:
        dimension = int(dimension_str)
        logger.info(f"Using embedding dimension: {dimension}")
        return dimension
    except ValueError:
        logger.warning(f"Invalid OPENAI_EMBEDDING_DIMENSION value: '{dimension_str}'. Using default: {DEFAULT_OPENAI_EMBEDDING_DIMENSION}")
        return DEFAULT_OPENAI_EMBEDDING_DIMENSION 