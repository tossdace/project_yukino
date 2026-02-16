import requests
import yaml
import soundfile as sf
import sounddevice as sd

from pathlib import Path


# -------------------------
# Load Config
# -------------------------

CONFIG_PATH = Path("character_files") / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    char_config = yaml.safe_load(f)

SOVITS_CFG = char_config["sovits_ping_config"]
SERVER_URL = SOVITS_CFG.get("server_url", "http://127.0.0.1:9880/tts")


# -------------------------
# Audio Playback
# -------------------------

def play_audio(path):
    try:
        data, samplerate = sf.read(path)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"[AUDIO ERROR] {e}")


# -------------------------
# TTS Generation
# -------------------------

def sovits_gen(text: str, output_path="output.wav"):

    output_path = Path(output_path)

    payload = {
        "text": text,
        "text_lang": SOVITS_CFG["text_lang"],
        "ref_audio_path": SOVITS_CFG["ref_audio_path"],
        "prompt_text": SOVITS_CFG["prompt_text"],
        "prompt_lang": SOVITS_CFG["prompt_lang"],
    }

    try:
        response = requests.post(
            SERVER_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        # Validate audio response
        if "audio" not in response.headers.get("Content-Type", ""):
            print("[TTS ERROR] Server did not return audio.")
            print(response.text[:200])
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path

    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return None


# -------------------------
# Test
# -------------------------

if __name__ == "__main__":
    import time

    start = time.time()

    path = sovits_gen(
        "If you hear this, the system is working.",
        "output.wav"
    )

    if path:
        play_audio(path)

    print(f"Elapsed: {time.time() - start:.2f}s")
