import json
from openai import OpenAI

from process.common.runtime_config import load_character_config, resolve_project_path


# -------------------------
# Load Config
# -------------------------

char_config = load_character_config()

API_KEY = char_config["OPENAI_API_KEY"]
MODEL = char_config["model"]
HISTORY_FILE = resolve_project_path(char_config["history_file"])

SYSTEM_PROMPT_TEXT = char_config["presets"]["default"]["system_prompt"]

# Limit conversation size
MAX_HISTORY_MESSAGES = 20


# -------------------------
# OpenAI Client
# -------------------------

_client = None
_history_cache = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY)
    return _client


# -------------------------
# History Management
# -------------------------

def _default_history():
    return [{
        "role": "system",
        "content": SYSTEM_PROMPT_TEXT
    }]


def _normalize_history(history):
    if not isinstance(history, list):
        return _default_history()

    sanitized_history = [
        message for message in history
        if isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
    ]

    if sanitized_history and sanitized_history[0]["role"] == "system":
        sanitized_history[0] = {
            "role": "system",
            "content": SYSTEM_PROMPT_TEXT,
        }
        return sanitized_history

    return _default_history() + sanitized_history


def load_history(force_reload: bool = False):
    global _history_cache

    if _history_cache is not None and not force_reload:
        return [message.copy() for message in _history_cache]

    if HISTORY_FILE.exists():
        try:
            with HISTORY_FILE.open("r", encoding="utf-8") as file:
                _history_cache = _normalize_history(json.load(file))
                return [message.copy() for message in _history_cache]
        except Exception:
            pass

    _history_cache = _default_history()
    return [message.copy() for message in _history_cache]


def save_history(history):
    global _history_cache

    history = trim_history(_normalize_history(history))
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)

    _history_cache = [message.copy() for message in history]


def trim_history(history):
    """
    Prevent infinite growth.
    Keeps system + last messages.
    """
    if len(history) > MAX_HISTORY_MESSAGES:
        system = history[0]
        retained_messages = MAX_HISTORY_MESSAGES - 1
        history = [system] + history[-retained_messages:]
    return history


# -------------------------
# LLM Call
# -------------------------

def call_llm(messages):
    client = get_client()

    response = client.responses.create(
        model=MODEL,
        input=messages,
        temperature=0.9,
        max_output_tokens=1024,
    )

    return response.output_text


# -------------------------
# Public Function
# -------------------------

def llm_response(user_input: str) -> str:
    cleaned_input = user_input.strip()
    if not cleaned_input:
        return ""

    history = load_history()

    history.append({
        "role": "user",
        "content": cleaned_input
    })

    history = trim_history(history)

    try:
        ai_text = call_llm(history)
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "Sorry, I had trouble thinking just now."

    history.append({
        "role": "assistant",
        "content": ai_text
    })

    save_history(history)

    return ai_text


# -------------------------
# Test
# -------------------------

if __name__ == "__main__":
    while True:
        text = input("You: ")
        print("AI:", llm_response(text))
