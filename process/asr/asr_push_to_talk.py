from pathlib import Path
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    def __init__(
        self,
        samplerate: int = 16000,
        device=None,
        blocksize: int = 1024,
    ):
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self._chunks = []
        self._lock = threading.Lock()
        self._stream = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _capture_chunk(self, indata, frames, time_info, status):
        if status:
            print(f"[ASR AUDIO] {status}")

        with self._lock:
            self._chunks.append(indata.copy())

    def start(self):
        if self.is_recording:
            raise RuntimeError("Recording is already in progress.")

        with self._lock:
            self._chunks = []

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=self.blocksize,
            callback=self._capture_chunk,
        )
        self._stream.start()

    def stop(self, output_file="recording.wav"):
        if not self.is_recording:
            raise RuntimeError("Recording has not been started.")

        stream = self._stream
        self._stream = None

        stream.stop()
        stream.close()

        with self._lock:
            recorded_chunks = self._chunks
            self._chunks = []

        if not recorded_chunks:
            return None

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            output_path.unlink()

        audio = np.concatenate(recorded_chunks, axis=0)
        sf.write(output_path, audio, self.samplerate)
        return output_path


def transcribe_audio(model, audio_path, beam_size: int = 5) -> str:
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text
    ).strip()


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

    recorder = AudioRecorder(
        samplerate=samplerate,
        device=device,
        blocksize=blocksize,
    )

    print("Press ENTER to start recording...")
    input()

    print("Recording... Press ENTER to stop")
    recorder.start()
    input()

    print("Saving audio...")
    output_path = recorder.stop(output_file)
    if not output_path:
        return ""

    print("Transcribing...")
    transcription = transcribe_audio(model, output_path)
    print(f"Transcription: {transcription}")

    return transcription
