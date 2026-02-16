import yaml
import json
import os
from pathlib import Path
from openai import OpenAI


# -------------------------
# Load Config
# -------------------------

CONFIG_PATH = Path("character_files") / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    char_config = yaml.safe_load(f)

API_KEY = char_config["OPENAI_API_KEY"]
MODEL = char_config["model"]
HISTORY_FILE = Path(char_config["history_file"])

SYSTEM_PROMPT_TEXT = char_config["presets"]["default"]["system_prompt"]

# Limit conversation size
MAX_HISTORY_MESSAGES = 20


# -------------------------
# OpenAI Client
# -------------------------

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY)
    return _client


# -------------------------
# History Management
# -------------------------

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Default history with system prompt
    return [{
        "role": "system",
        "content": SYSTEM_PROMPT_TEXT
    }]


def save_history(history):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def trim_history(history):
    """
    Prevent infinite growth.
    Keeps system + last messages.
    """
    if len(history) > MAX_HISTORY_MESSAGES:
        system = history[0]
        history = [system] + history[-MAX_HISTORY_MESSAGES:]
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

    history = load_history()

    history.append({
        "role": "user",
        "content": user_input
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
