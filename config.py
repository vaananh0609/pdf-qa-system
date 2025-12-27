import os
import secrets
import logging
from dotenv import load_dotenv

load_dotenv()


def _split_csv_env(value: str | None) -> list[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(',')]
    return [p for p in parts if p]


class Config:
    # Flask secret key (REQUIRED for stable sessions). If not provided, we generate a random
    # value for local dev to avoid committing a real secret into the repo.
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        SECRET_KEY = secrets.token_hex(32)
        logging.warning('SECRET_KEY chưa được set; dùng secret ngẫu nhiên (chỉ phù hợp dev).')

    # Database
    # Default to local MongoDB to avoid leaking any cloud credentials in the repo.
    MONGODB_URI = os.environ.get('MONGODB_URI') or 'mongodb://localhost:27017'

    # Admin login (should be set via env in production)
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME') or 'admin'
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')  # no default on purpose

    # AI keys
    # Prefer GEMINI_API_KEYS as comma-separated list; fallback to *_PRIMARY/*_SECONDARY...
    _gemini_keys = _split_csv_env(os.environ.get('GEMINI_API_KEYS'))
    if not _gemini_keys:
        _gemini_keys = [
            os.environ.get('GEMINI_API_KEY_PRIMARY') or '',
            os.environ.get('GEMINI_API_KEY_SECONDARY') or '',
            os.environ.get('GEMINI_API_KEY_TERTIARY') or '',
            os.environ.get('GEMINI_API_KEY_4') or '',
        ]
        _gemini_keys = [k for k in _gemini_keys if k]
    GEMINI_API_KEYS = _gemini_keys
    GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else None

    # Vintern Embedding API URL (from Colab server)
    VINTERN_API_URL = os.environ.get('VINTERN_API_URL')

    # Groq API key (optional fallback)
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

    # Uploads
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # Whether to attempt MongoDB connection at application startup (fail-fast).
    # Set to True in production if you want the app to raise configuration errors early.
    DB_CONNECT_ON_START = os.environ.get('DB_CONNECT_ON_START', 'false').lower() in ('1', 'true', 'yes')

    # Ensure uploads folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
