import argparse
import uuid
from pathlib import Path

from faster_whisper import WhisperModel

from process.asr.asr_push_to_talk import record_and_transcribe
from process.common.runtime_config import AUDIO_DIR
from process.llm.llm_scr import llm_response
from process.tts.sovits_ping import play_audio, sovits_gen


EXIT_COMMANDS = {"exit", "quit", "stop"}
WHISPER_MODEL_NAME = "base.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"


def cleanup_generated_audio(audio_dir: Path):
    """
    Remove only generated TTS files.
    Keeps conversation.wav safe.
    """
    for audio_path in audio_dir.glob("output_*.wav"):
        try:
            audio_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[WARN] Could not delete {audio_path}: {exc}")


def create_whisper_model() -> WhisperModel:
    print("Loading Whisper model (CPU optimized)...")
    return WhisperModel(
        WHISPER_MODEL_NAME,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )


def play_response(ai_response: str, audio_dir: Path):
    output_path = audio_dir / f"output_{uuid.uuid4().hex}.wav"

    print("Generating speech...")
    generated_path = sovits_gen(ai_response, output_path)

    if not generated_path:
        print("[ERROR] TTS generation failed.\n")
        return

    print("Playing audio...\n")
    try:
        play_audio(generated_path)
    finally:
        try:
            Path(generated_path).unlink(missing_ok=True)
        except Exception as exc:
            print(f"[WARN] Could not delete {generated_path}: {exc}")


def run_cli_chat():
    print("\n========= Starting Voice Chat =========\n")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_generated_audio(AUDIO_DIR)

    whisper_model = create_whisper_model()
    conversation_path = AUDIO_DIR / "conversation.wav"

    print("System ready.\n")

    while True:
        try:
            print("Listening...")
            user_text = record_and_transcribe(whisper_model, conversation_path)

            if not user_text or not user_text.strip():
                print("No speech detected.\n")
                continue

            user_text = user_text.strip()
            print(f"\nUser: {user_text}")

            if user_text.casefold() in EXIT_COMMANDS:
                print("Exiting chat...")
                break

            print("Generating response...")
            ai_response = llm_response(user_text)

            if not ai_response:
                print("LLM returned empty response.\n")
                continue

            print(f"AI: {ai_response}")
            play_response(ai_response, AUDIO_DIR)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as exc:
            print(f"\n[ERROR] {exc}\n")


def main():
    parser = argparse.ArgumentParser(description="Yukino voice assistant")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="run the original terminal interface instead of the desktop UI",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli_chat()
        return

    from process.ui.desktop_app import launch_desktop_ui

    launch_desktop_ui()


if __name__ == "__main__":
    main()
