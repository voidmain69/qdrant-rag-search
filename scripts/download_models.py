"""Pre-warm the FastEmbed model cache (e.g. into the docker model_cache volume).

Usage: uv run python scripts/download_models.py
Respects FASTEMBED_CACHE_PATH.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.embedding import EmbeddingService  # noqa: E402


def main() -> None:
    settings = get_settings()
    print(f"Downloading dense={settings.dense_model} sparse={settings.sparse_model} ...")
    EmbeddingService(settings)  # constructor downloads + warms up
    print("Models cached.")


if __name__ == "__main__":
    main()
