"""
Env vars and constants for the growth experimentation copilot.
"""

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1000

# Statistics / experiment design
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
MIN_RUNTIME_DAYS = 7
MAX_RUNTIME_DAYS = 30
NOVELTY_WINDOW_DAYS = 3
SRM_THRESHOLD = 0.01

# Power analysis defaults (designer)
DEFAULT_BASE_CONVERSION_RATE = 0.12
DEFAULT_MDE = 0.05  # minimum detectable effect
