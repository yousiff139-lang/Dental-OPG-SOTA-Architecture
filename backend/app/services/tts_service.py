import os
import re
import hashlib
import logging
import httpx
from pathlib import Path
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Persistent cache directory
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "audio_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for fast access
_audio_cache: dict[str, bytes] = {}

def clean_text_for_speech(text: str) -> str:
    # Remove emojis
    cleaned = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Remove markdown asterisks, underscores, backticks, hashes
    cleaned = re.sub(r'[*_#`~]', '', cleaned)
    # Pronounce XRayBo naturally
    cleaned = re.sub(r'\bXRayBo\b', 'X-Ray Bo', cleaned, flags=re.IGNORECASE)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_cache_path(text: str) -> Path:
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{text_hash}.mp3"

async def generate_speech_fish_audio(text: str) -> bytes | None:
    """
    Generate speech using Fish Audio TTS API with a calm natural human voice.
    Checks memory cache and disk cache before calling API.
    """
    cleaned_text = clean_text_for_speech(text)
    if not cleaned_text:
        return None

    # 1. Check in-memory cache
    if cleaned_text in _audio_cache:
        return _audio_cache[cleaned_text]

    # 2. Check disk cache
    cache_file = get_cache_path(cleaned_text)
    if cache_file.exists():
        try:
            audio_bytes = cache_file.read_bytes()
            _audio_cache[cleaned_text] = audio_bytes
            return audio_bytes
        except Exception as e:
            logger.warning(f"Error reading disk cache: {e}")

    # 3. Call Fish Audio API
    settings = get_settings()
    api_key = settings.fish_audio_api_key
    if not api_key:
        logger.warning("No Fish Audio API key configured")
        return None

    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "model": "s2.1-pro-free",
    }
    payload = {
        "text": cleaned_text,
        "format": "mp3",
        "reference_id": "0b74ead073f2474a904f69033535b98e", # Calm Normal human voice
        "latency": "normal",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                audio_bytes = response.content
                # Save to disk cache
                try:
                    cache_file.write_bytes(audio_bytes)
                except Exception as e:
                    logger.warning(f"Error writing to disk cache: {e}")
                
                # Save to memory cache
                _audio_cache[cleaned_text] = audio_bytes
                return audio_bytes
            else:
                logger.error(f"Fish Audio error {response.status_code}: {response.text}")
                return None
    except Exception as e:
        logger.error(f"Error calling Fish Audio: {e}")
        return None
