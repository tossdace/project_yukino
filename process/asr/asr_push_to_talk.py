import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path


def record_and_transcribe(
    model,
    output_file="recording.wav",
    samplerate=16000,
    device=None
):
    """
    Push-to-talk recording:
    ENTER -> start
    ENTER -> stop
    Returns transcribed text.
    """

    output_path = Path(output_file)

    if output_path.exists():
        output_path.unlink()

    print("Press ENTER to start recording...")
    input()

    print("🔴 Recording... Press ENTER to stop")

    recording = []
    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device
    )

    with stream:
        while True:
            chunk, _ = stream.read(1024)
            recording.append(chunk)

            if sd.wait(0):
                pass

            # Non-blocking stop check
            if os.name == "nt":
                import msvcrt
                if msvcrt.kbhit():
                    msvcrt.getch()
                    break
            else:
                # fallback
                break

    audio = np.concatenate(recording, axis=0)

    print("⏹️  Saving audio...")
    sf.write(output_path, audio, samplerate)

    print("🎯 Transcribing...")

    segments, _ = model.transcribe(
        str(output_path),
        beam_size=5
    )

    transcription = " ".join(seg.text for seg in segments).strip()

    print(f"Transcription: {transcription}")

    return transcription
