from faster_whisper import WhisperModel

from process.asr.asr_push_to_talk import record_and_transcribe
from process.llm.llm_scr import llm_response
from process.tts.sovits_ping import sovits_gen, play_audio

from pathlib import Path
import uuid


# ----------------------------
# Utility
# ----------------------------

def cleanup_audio_files(audio_dir: Path):
    """
    Remove only generated TTS files.
    Keeps conversation.wav safe.
    """
    for fp in audio_dir.glob("output_*.wav"):
        try:
            fp.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Could not delete {fp}: {e}")


# ----------------------------
# Startup
# ----------------------------

print("\n========= Starting Voice Chat =========\n")

audio_dir = Path("audio")
audio_dir.mkdir(parents=True, exist_ok=True)

print("Loading Whisper model (CPU optimized)...")

whisper_model = WhisperModel(
    "base.en",      # use small.en if you want better accuracy
    device="cpu",
    compute_type="int8"
)

print("System ready.\n")


# ----------------------------
# Main Loop
# ----------------------------

while True:
    try:
        conversation_path = audio_dir / "conversation.wav"

        print("Listening...")
        user_text = record_and_transcribe(
            whisper_model,
            conversation_path
        )

        if not user_text or not user_text.strip():
            print("No speech detected.\n")
            continue

        user_text = user_text.strip()
        print(f"\nUser: {user_text}")

        # Exit condition
        if user_text.lower() in ["exit", "quit", "stop"]:
            print("Exiting chat...")
            break

        # ---------------- LLM ----------------
        print("Generating response...")
        ai_response = llm_response(user_text)

        if not ai_response:
            print("LLM returned empty response.\n")
            continue

        print(f"AI: {ai_response}")

        # ---------------- TTS ----------------
        uid = uuid.uuid4().hex
        output_path = audio_dir / f"output_{uid}.wav"

        print("Generating speech...")
        generated_path = sovits_gen(ai_response, output_path)

        if not generated_path:
            print("[ERROR] TTS generation failed.\n")
            continue

        print("Playing audio...\n")
        play_audio(generated_path)

        # Cleanup generated audio (not conversation file)
        cleanup_audio_files(audio_dir)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        break

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
