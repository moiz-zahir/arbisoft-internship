import logging

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
PING_TIMEOUT_SECONDS = 2.0


def is_ollama_available(url: str = OLLAMA_URL, timeout: float = PING_TIMEOUT_SECONDS) -> bool:
    """
    Checks whether a local Ollama server is reachable right now.

    This is a live check, not a cached setting, because Ollama can be
    started or stopped between runs (or even mid-batch) - trusting a stale
    "it was up earlier" would mean every categorization in a batch could
    fail the same way instead of routing around the outage before the batch
    even starts.
    """
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_available_backend() -> str:
    """Returns "local" if Ollama is reachable right now, otherwise "cloud"."""
    return "local" if is_ollama_available() else "cloud"


def log_model_used(description: str, route: str) -> None:
    """Records which backend actually categorized a given transaction."""
    logger.info("Categorized %r using the %s model", description, route)
