import queue
import threading
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext

from faster_whisper import WhisperModel

from process.asr.asr_push_to_talk import AudioRecorder, transcribe_audio
from process.common.runtime_config import AUDIO_DIR
from process.llm.llm_scr import MODEL as OPENAI_MODEL
from process.llm.llm_scr import load_history, llm_response
from process.tts.sovits_ping import SERVER_URL, play_audio, sovits_gen


WHISPER_MODEL_NAME = "base.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

PALETTE = {
    "bg": "#09111f",
    "bg_alt": "#111c31",
    "panel": "#13213a",
    "panel_alt": "#172844",
    "panel_soft": "#1b2f4f",
    "ink": "#f3eee4",
    "muted": "#92a5c6",
    "accent": "#ff7a59",
    "accent_soft": "#ffb79a",
    "teal": "#88dccf",
    "gold": "#f6c76b",
    "user": "#8fd7ff",
    "assistant": "#ffb87a",
    "system": "#9db1d8",
    "danger": "#ff8f8f",
}


def cleanup_generated_audio(audio_dir: Path):
    for audio_path in audio_dir.glob("output_*.wav"):
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


def create_whisper_model() -> WhisperModel:
    return WhisperModel(
        WHISPER_MODEL_NAME,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )


class YukinoDesktopApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Yukino Voice Studio")
        self.geometry("1240x860")
        self.minsize(980, 720)
        self.configure(bg=PALETTE["bg"])

        self.ui_queue = queue.Queue()
        self.recorder = AudioRecorder()
        self.whisper_model = None
        self.is_busy = False
        self.is_recording = False
        self.conversation_path = AUDIO_DIR / "conversation.wav"

        self.status_var = tk.StringVar(value="Booting voice systems...")
        self.subtitle_var = tk.StringVar(value="Loading the local speech stack.")
        self.voice_state_var = tk.StringVar(value="Whisper loading")
        self.memory_state_var = tk.StringVar(value="Memory available")
        self.reply_state_var = tk.StringVar(value="Reply lane idle")

        self._build_layout()
        self._configure_transcript_tags()
        self._hydrate_history_view()
        self._refresh_controls()

        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        cleanup_generated_audio(AUDIO_DIR)

        self.after(100, self._drain_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._handle_close)

        threading.Thread(target=self._bootstrap_services, daemon=True).start()

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header_frame = tk.Frame(
            self,
            bg=PALETTE["panel"],
            padx=28,
            pady=24,
            highlightthickness=1,
            highlightbackground="#233a60",
        )
        self.header_frame.grid(row=0, column=0, sticky="nsew", padx=24, pady=(24, 12))
        self.header_frame.grid_columnconfigure(0, weight=3)
        self.header_frame.grid_columnconfigure(1, weight=2)

        title_block = tk.Frame(self.header_frame, bg=PALETTE["panel"])
        title_block.grid(row=0, column=0, sticky="nsew")

        tk.Label(
            title_block,
            text="VOICE STUDIO",
            bg=PALETTE["panel"],
            fg=PALETTE["accent_soft"],
            font=("Bahnschrift SemiBold", 11),
        ).pack(anchor="w")

        tk.Label(
            title_block,
            text="Yukino",
            bg=PALETTE["panel"],
            fg=PALETTE["ink"],
            font=("Constantia", 34, "bold"),
        ).pack(anchor="w", pady=(8, 4))

        tk.Label(
            title_block,
            textvariable=self.subtitle_var,
            bg=PALETTE["panel"],
            fg=PALETTE["muted"],
            font=("Corbel", 13),
            wraplength=520,
            justify="left",
        ).pack(anchor="w")

        status_block = tk.Frame(self.header_frame, bg=PALETTE["panel"])
        status_block.grid(row=0, column=1, sticky="nsew", padx=(24, 0))
        status_block.grid_columnconfigure(0, weight=1)
        status_block.grid_columnconfigure(1, weight=1)

        self.status_pill = tk.Label(
            status_block,
            textvariable=self.status_var,
            bg=PALETTE["accent"],
            fg=PALETTE["bg"],
            padx=18,
            pady=8,
            font=("Bahnschrift SemiBold", 11),
        )
        self.status_pill.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        self.voice_card = self._build_info_card(
            status_block,
            title="Voice Lane",
            value_var=self.voice_state_var,
            accent=PALETTE["teal"],
        )
        self.voice_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        self.reply_card = self._build_info_card(
            status_block,
            title="Reply Lane",
            value_var=self.reply_state_var,
            accent=PALETTE["gold"],
        )
        self.reply_card.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        self.body_frame = tk.Frame(self, bg=PALETTE["bg"])
        self.body_frame.grid(row=1, column=0, sticky="nsew", padx=24, pady=12)
        self.body_frame.grid_columnconfigure(0, weight=3)
        self.body_frame.grid_columnconfigure(1, weight=2)
        self.body_frame.grid_rowconfigure(0, weight=1)

        transcript_card = tk.Frame(
            self.body_frame,
            bg=PALETTE["bg_alt"],
            highlightthickness=1,
            highlightbackground="#203352",
        )
        transcript_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        transcript_card.grid_rowconfigure(1, weight=1)
        transcript_card.grid_columnconfigure(0, weight=1)

        transcript_header = tk.Frame(transcript_card, bg=PALETTE["bg_alt"], padx=22, pady=18)
        transcript_header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            transcript_header,
            text="Conversation Deck",
            bg=PALETTE["bg_alt"],
            fg=PALETTE["ink"],
            font=("Constantia", 21, "bold"),
        ).pack(anchor="w")
        tk.Label(
            transcript_header,
            text="Type naturally or drive the session with voice capture.",
            bg=PALETTE["bg_alt"],
            fg=PALETTE["muted"],
            font=("Corbel", 12),
        ).pack(anchor="w", pady=(4, 0))

        self.transcript = scrolledtext.ScrolledText(
            transcript_card,
            wrap="word",
            bg=PALETTE["bg_alt"],
            fg=PALETTE["ink"],
            insertbackground=PALETTE["ink"],
            relief="flat",
            padx=24,
            pady=10,
            font=("Corbel", 12),
            spacing1=4,
            spacing2=2,
            spacing3=10,
        )
        self.transcript.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 8))
        self.transcript.configure(state="disabled")

        sidebar = tk.Frame(self.body_frame, bg=PALETTE["bg"])
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(0, weight=1)
        sidebar.grid_rowconfigure(1, weight=1)

        control_card = tk.Frame(
            sidebar,
            bg=PALETTE["panel_alt"],
            padx=22,
            pady=22,
            highlightthickness=1,
            highlightbackground="#2c456d",
        )
        control_card.grid(row=0, column=0, sticky="nsew", pady=(0, 12))

        tk.Label(
            control_card,
            text="Voice Controls",
            bg=PALETTE["panel_alt"],
            fg=PALETTE["ink"],
            font=("Constantia", 20, "bold"),
        ).pack(anchor="w")

        tk.Label(
            control_card,
            text="Use the mic buttons when you want a spoken, full round-trip reply.",
            bg=PALETTE["panel_alt"],
            fg=PALETTE["muted"],
            font=("Corbel", 12),
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(6, 18))

        self.record_button = tk.Button(
            control_card,
            text="Start Voice Capture",
            command=self._start_recording,
            bg=PALETTE["accent"],
            fg=PALETTE["bg"],
            activebackground="#ff8d70",
            activeforeground=PALETTE["bg"],
            relief="flat",
            padx=16,
            pady=12,
            font=("Bahnschrift SemiBold", 12),
            cursor="hand2",
        )
        self.record_button.pack(fill="x")

        self.stop_button = tk.Button(
            control_card,
            text="Stop And Send",
            command=self._stop_recording,
            bg=PALETTE["teal"],
            fg=PALETTE["bg"],
            activebackground="#a6f0e4",
            activeforeground=PALETTE["bg"],
            relief="flat",
            padx=16,
            pady=12,
            font=("Bahnschrift SemiBold", 12),
            cursor="hand2",
        )
        self.stop_button.pack(fill="x", pady=(12, 0))

        self.session_hint = tk.Label(
            control_card,
            text="The transcript below persists across replies through the existing history file.",
            bg=PALETTE["panel_alt"],
            fg=PALETTE["muted"],
            font=("Corbel", 11),
            wraplength=320,
            justify="left",
        )
        self.session_hint.pack(anchor="w", pady=(16, 0))

        info_card = tk.Frame(
            sidebar,
            bg=PALETTE["panel_soft"],
            padx=22,
            pady=22,
            highlightthickness=1,
            highlightbackground="#35527f",
        )
        info_card.grid(row=1, column=0, sticky="nsew")

        tk.Label(
            info_card,
            text="Session Signal",
            bg=PALETTE["panel_soft"],
            fg=PALETTE["ink"],
            font=("Constantia", 20, "bold"),
        ).pack(anchor="w")

        self._build_stat_line(info_card, "OpenAI model", OPENAI_MODEL)
        self._build_stat_line(info_card, "Whisper profile", f"{WHISPER_MODEL_NAME} / {WHISPER_COMPUTE_TYPE}")
        self._build_stat_line(info_card, "TTS server", SERVER_URL)

        tk.Label(
            info_card,
            textvariable=self.memory_state_var,
            bg=PALETTE["panel_soft"],
            fg=PALETTE["system"],
            font=("Corbel", 11),
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(18, 0))

        composer_card = tk.Frame(
            self,
            bg=PALETTE["panel"],
            padx=24,
            pady=18,
            highlightthickness=1,
            highlightbackground="#22395d",
        )
        composer_card.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 24))
        composer_card.grid_columnconfigure(0, weight=1)

        tk.Label(
            composer_card,
            text="Text Composer",
            bg=PALETTE["panel"],
            fg=PALETTE["ink"],
            font=("Constantia", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            composer_card,
            text="Press Enter to send. Use Shift+Enter for a new line.",
            bg=PALETTE["panel"],
            fg=PALETTE["muted"],
            font=("Corbel", 11),
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        self.input_box = tk.Text(
            composer_card,
            height=4,
            wrap="word",
            bg="#0f192c",
            fg=PALETTE["ink"],
            insertbackground=PALETTE["ink"],
            relief="flat",
            padx=16,
            pady=14,
            font=("Corbel", 12),
        )
        self.input_box.grid(row=2, column=0, sticky="ew", padx=(0, 14))
        self.input_box.bind("<Return>", self._handle_return_key)

        send_button_row = tk.Frame(composer_card, bg=PALETTE["panel"])
        send_button_row.grid(row=2, column=1, sticky="ns")

        self.send_button = tk.Button(
            send_button_row,
            text="Send Message",
            command=self._send_text_message,
            bg=PALETTE["gold"],
            fg=PALETTE["bg"],
            activebackground="#ffd98b",
            activeforeground=PALETTE["bg"],
            relief="flat",
            padx=18,
            pady=12,
            font=("Bahnschrift SemiBold", 12),
            cursor="hand2",
        )
        self.send_button.pack(fill="x")

        self.clear_button = tk.Button(
            send_button_row,
            text="Clear Draft",
            command=self._clear_input,
            bg=PALETTE["panel_soft"],
            fg=PALETTE["ink"],
            activebackground="#29436c",
            activeforeground=PALETTE["ink"],
            relief="flat",
            padx=18,
            pady=12,
            font=("Bahnschrift SemiBold", 12),
            cursor="hand2",
        )
        self.clear_button.pack(fill="x", pady=(12, 0))

    def _build_info_card(self, parent, title: str, value_var: tk.StringVar, accent: str):
        card = tk.Frame(
            parent,
            bg=PALETTE["panel_alt"],
            padx=16,
            pady=14,
            highlightthickness=1,
            highlightbackground="#274167",
        )

        tk.Label(
            card,
            text=title,
            bg=PALETTE["panel_alt"],
            fg=accent,
            font=("Bahnschrift SemiBold", 10),
        ).pack(anchor="w")

        tk.Label(
            card,
            textvariable=value_var,
            bg=PALETTE["panel_alt"],
            fg=PALETTE["ink"],
            font=("Corbel", 12),
            wraplength=180,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        return card

    def _build_stat_line(self, parent, label: str, value: str):
        line = tk.Frame(parent, bg=PALETTE["panel_soft"])
        line.pack(fill="x", pady=(16, 0))

        tk.Label(
            line,
            text=label,
            bg=PALETTE["panel_soft"],
            fg=PALETTE["muted"],
            font=("Bahnschrift SemiBold", 10),
        ).pack(anchor="w")

        tk.Label(
            line,
            text=value,
            bg=PALETTE["panel_soft"],
            fg=PALETTE["ink"],
            font=("Consolas", 10),
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

    def _configure_transcript_tags(self):
        self.transcript.tag_configure(
            "speaker_user",
            foreground=PALETTE["user"],
            font=("Bahnschrift SemiBold", 11),
            spacing1=10,
        )
        self.transcript.tag_configure(
            "speaker_assistant",
            foreground=PALETTE["assistant"],
            font=("Bahnschrift SemiBold", 11),
            spacing1=10,
        )
        self.transcript.tag_configure(
            "speaker_system",
            foreground=PALETTE["system"],
            font=("Bahnschrift SemiBold", 11),
            spacing1=10,
        )
        self.transcript.tag_configure(
            "body",
            foreground=PALETTE["ink"],
            font=("Corbel", 12),
            lmargin1=0,
            lmargin2=0,
        )

    def _hydrate_history_view(self):
        try:
            history = load_history(force_reload=True)
        except Exception:
            history = []

        visible_messages = [
            message for message in history
            if message.get("role") in {"user", "assistant"}
        ]

        turns_loaded = len(visible_messages)
        if turns_loaded:
            self.memory_state_var.set(f"Loaded {turns_loaded} saved messages into the deck.")
        else:
            self.memory_state_var.set("No saved turns yet. This session will start fresh.")

        if not visible_messages:
            self._append_message(
                "system",
                "The room is live. Start with text or use the voice controls when Whisper finishes loading.",
            )
            return

        for message in visible_messages:
            self._append_message(message["role"], message["content"])

    def _append_message(self, role: str, text: str):
        speaker = {
            "user": "YOU",
            "assistant": "YUKINO",
            "system": "SYSTEM",
        }.get(role, "SYSTEM")

        speaker_tag = {
            "user": "speaker_user",
            "assistant": "speaker_assistant",
            "system": "speaker_system",
        }.get(role, "speaker_system")

        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"{speaker}\n", speaker_tag)
        self.transcript.insert("end", f"{text.strip()}\n\n", "body")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _set_status(self, status: str, subtitle: str, *, pill_color: str):
        self.status_var.set(status)
        self.subtitle_var.set(subtitle)
        self.status_pill.configure(bg=pill_color)

    def _set_idle_voice_state(self):
        if self.whisper_model is not None:
            self.voice_state_var.set("Voice capture armed")
        else:
            self.voice_state_var.set("Whisper unavailable")

    def _refresh_controls(self):
        allow_text = not self.is_busy and not self.is_recording
        allow_record = self.whisper_model is not None and not self.is_busy and not self.is_recording
        allow_stop = self.is_recording

        self.send_button.configure(state="normal" if allow_text else "disabled")
        self.record_button.configure(state="normal" if allow_record else "disabled")
        self.stop_button.configure(state="normal" if allow_stop else "disabled")

    def _handle_return_key(self, event):
        if event.state & 0x1:
            return None

        self._send_text_message()
        return "break"

    def _clear_input(self):
        self.input_box.delete("1.0", "end")

    def _bootstrap_services(self):
        self.ui_queue.put((
            "status",
            "Loading speech model...",
            "Warming up Faster-Whisper for voice capture.",
            PALETTE["accent"],
        ))

        try:
            model = create_whisper_model()
        except Exception as exc:
            self.ui_queue.put(("error", f"Whisper failed to load: {exc}"))
            return

        self.ui_queue.put(("model_ready", model))

    def _start_recording(self):
        if self.whisper_model is None or self.is_busy or self.is_recording:
            return

        try:
            self.recorder.start()
        except Exception as exc:
            self._append_message("system", f"Microphone error: {exc}")
            self._set_status(
                "Microphone unavailable",
                "Voice capture could not start. Check your input device.",
                pill_color=PALETTE["danger"],
            )
            return

        self.is_recording = True
        self.voice_state_var.set("Recording live input")
        self.reply_state_var.set("Waiting for your stop command")
        self._set_status(
            "Recording...",
            "Capture in progress. Press Stop And Send when you are ready.",
            pill_color=PALETTE["teal"],
        )
        self._refresh_controls()

    def _stop_recording(self):
        if not self.is_recording:
            return

        try:
            audio_path = self.recorder.stop(self.conversation_path)
        except Exception as exc:
            self.is_recording = False
            self._refresh_controls()
            self._append_message("system", f"Recording stop failed: {exc}")
            self._set_status(
                "Recording interrupted",
                "The microphone stopped unexpectedly.",
                pill_color=PALETTE["danger"],
            )
            return

        self.is_recording = False

        if not audio_path:
            self.voice_state_var.set("No audio captured")
            self.reply_state_var.set("Reply lane idle")
            self._set_status(
                "No voice detected",
                "Try another recording and make sure the mic is active.",
                pill_color=PALETTE["gold"],
            )
            self._refresh_controls()
            return

        self.is_busy = True
        self.voice_state_var.set("Transcribing latest capture")
        self.reply_state_var.set("Preparing a spoken reply")
        self._set_status(
            "Transcribing...",
            "Converting the microphone capture into text.",
            pill_color=PALETTE["gold"],
        )
        self._refresh_controls()

        threading.Thread(
            target=self._transcribe_and_reply,
            args=(audio_path,),
            daemon=True,
        ).start()

    def _send_text_message(self):
        if self.is_busy or self.is_recording:
            return

        user_text = self.input_box.get("1.0", "end").strip()
        if not user_text:
            return

        self._clear_input()
        self._append_message("user", user_text)
        self.is_busy = True
        self.reply_state_var.set("Thinking through your message")
        self._set_status(
            "Thinking...",
            "Sending your message through the reasoning pipeline.",
            pill_color=PALETTE["accent"],
        )
        self._refresh_controls()

        threading.Thread(
            target=self._generate_and_play_reply,
            args=(user_text,),
            daemon=True,
        ).start()

    def _transcribe_and_reply(self, audio_path: Path):
        try:
            user_text = transcribe_audio(self.whisper_model, audio_path)
            if not user_text:
                self.ui_queue.put(("transcript_empty",))
                return

            self.ui_queue.put(("message", "user", user_text))
            self.ui_queue.put((
                "status",
                "Thinking...",
                "Voice transcribed. Building the reply now.",
                PALETTE["accent"],
            ))
            self.ui_queue.put(("reply_state", "Thinking through your voice message"))
            self._run_reply_pipeline(user_text)
        except Exception as exc:
            self.ui_queue.put(("error", f"Voice processing failed: {exc}"))
        finally:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _generate_and_play_reply(self, user_text: str):
        try:
            self._run_reply_pipeline(user_text)
        except Exception as exc:
            self.ui_queue.put(("error", f"Message processing failed: {exc}"))

    def _run_reply_pipeline(self, user_text: str):
        ai_response = llm_response(user_text)
        if not ai_response:
            self.ui_queue.put(("reply_failed", "The assistant returned an empty response."))
            return

        self.ui_queue.put(("message", "assistant", ai_response))
        self.ui_queue.put((
            "status",
            "Speaking...",
            "The reply text is ready and audio playback is starting.",
            PALETTE["teal"],
        ))
        self.ui_queue.put(("reply_state", "Playing the generated response"))

        output_path = AUDIO_DIR / f"output_{uuid.uuid4().hex}.wav"
        generated_path = sovits_gen(ai_response, output_path)
        if not generated_path:
            self.ui_queue.put(("speech_failed",))
            return

        try:
            played = play_audio(generated_path)
            if not played:
                self.ui_queue.put(("speech_failed",))
                return
        finally:
            try:
                Path(generated_path).unlink(missing_ok=True)
            except Exception:
                pass

        self.ui_queue.put(("done",))

    def _drain_ui_queue(self):
        while True:
            try:
                event = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            kind = event[0]

            if kind == "model_ready":
                self.whisper_model = event[1]
                self._set_idle_voice_state()
                self.reply_state_var.set("Reply lane idle")
                self._set_status(
                    "Studio ready",
                    "Text and voice controls are ready for the next turn.",
                    pill_color=PALETTE["teal"],
                )
                self._refresh_controls()
                continue

            if kind == "message":
                _, role, text = event
                self._append_message(role, text)
                continue

            if kind == "status":
                _, status, subtitle, pill_color = event
                self._set_status(status, subtitle, pill_color=pill_color)
                continue

            if kind == "reply_state":
                self.reply_state_var.set(event[1])
                continue

            if kind == "transcript_empty":
                self.is_busy = False
                self.voice_state_var.set("No words found in latest capture")
                self.reply_state_var.set("Reply lane idle")
                self._append_message("system", "No speech was detected in that recording.")
                self._set_status(
                    "Nothing transcribed",
                    "Try another take or move closer to the microphone.",
                    pill_color=PALETTE["gold"],
                )
                self._refresh_controls()
                continue

            if kind == "reply_failed":
                self.is_busy = False
                self.reply_state_var.set("Reply lane idle")
                self._append_message("system", event[1])
                self._set_status(
                    "Reply unavailable",
                    "The assistant did not produce a response this turn.",
                    pill_color=PALETTE["danger"],
                )
                self._refresh_controls()
                continue

            if kind == "speech_failed":
                self.is_busy = False
                self.reply_state_var.set("Text reply ready, speech failed")
                self._append_message("system", "The text reply is ready, but audio playback failed.")
                self._set_status(
                    "Speech issue",
                    "Text is available in the transcript even though audio failed.",
                    pill_color=PALETTE["gold"],
                )
                self._refresh_controls()
                continue

            if kind == "error":
                self.is_busy = False
                self.is_recording = False
                self.voice_state_var.set("Voice lane needs attention")
                self.reply_state_var.set("Reply lane idle")
                self._append_message("system", event[1])
                self._set_status(
                    "Something broke",
                    "Check the transcript for the latest error message.",
                    pill_color=PALETTE["danger"],
                )
                self._refresh_controls()
                continue

            if kind == "done":
                self.is_busy = False
                self._set_idle_voice_state()
                self.reply_state_var.set("Reply lane idle")
                self._set_status(
                    "Studio ready",
                    "Waiting for the next text message or voice capture.",
                    pill_color=PALETTE["teal"],
                )
                self._refresh_controls()
                continue

        self.after(120, self._drain_ui_queue)

    def _handle_close(self):
        if self.is_recording:
            try:
                self.recorder.stop(self.conversation_path)
            except Exception:
                pass

        self.destroy()


def launch_desktop_ui():
    app = YukinoDesktopApp()
    app.mainloop()
