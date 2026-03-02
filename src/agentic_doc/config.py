"""Configuration: API key, model names, feature flags."""

import os


def get_gemini_api_key() -> str:
    """Get Gemini API key from Colab secrets or environment."""
    try:
        from google.colab import userdata
        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key
    except (ImportError, Exception):
        pass
    key = os.environ.get("gemini-api") or os.environ.get("GOOGLE_API_KEY")
    return key or ""


def load_config():
    """Load config from env; set GOOGLE_API_KEY for libs."""
    api_key = get_gemini_api_key()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    return {
        "GOOGLE_API_KEY": api_key,
        "AGENT_MODEL": os.environ.get("AGENT_MODEL", "gemini-2.0-flash"),
        "VISION_MODEL": os.environ.get("VISION_MODEL", "gemini-3-flash-preview"),
        "USE_LAYOUT_FALLBACK": os.environ.get("USE_LAYOUT_FALLBACK", "").lower() in ("1", "true", "yes"),
    }
