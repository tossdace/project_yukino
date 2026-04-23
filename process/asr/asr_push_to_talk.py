import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def _wait_for_stop(stop_event: threading.Event):
    input()
    stop_event.set()


def record_and_transcribe(
    model,
    output_file="recording.wav",
    samplerate=16000,
    device=None,
    blocksize=1024,
):
    """
    Push-to-talk recording:
    ENTER -> start
    ENTER -> stop
    Returns transcribed text.
    """

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    print("Press ENTER to start recording...")
    input()

    print("Recording... Press ENTER to stop")

    recording = []
    stop_event = threading.Event()
    stop_listener = threading.Thread(
        target=_wait_for_stop,
        args=(stop_event,),
        daemon=True,
    )
    stop_listener.start()

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device,
        blocksize=blocksize,
    )

    with stream:
        while not stop_event.is_set():
            chunk, _ = stream.read(blocksize)
            if chunk.size:
                recording.append(chunk.copy())

    if not recording:
        return ""

    audio = np.concatenate(recording, axis=0)

    print("Saving audio...")
    sf.write(output_path, audio, samplerate)

    print("Transcribing...")

    segments, _ = model.transcribe(
        str(output_path),
        beam_size=5
    )

    transcription = " ".join(
        seg.text.strip()
        for seg in segments
        if seg.text
    ).strip()

    print(f"Transcription: {transcription}")

    return transcription
