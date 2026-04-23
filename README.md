# ❄️ Yukino — Local AI Voice Assistant

**Yukino** is a modular voice-to-voice AI assistant that listens, thinks, and speaks using a custom cloned voice.

Built with:

* 🎤 Faster-Whisper (Speech Recognition)
* 🧠 OpenAI (Reasoning / Conversation)
* 🔊 GPT-SoVITS (Voice Synthesis)
* 🖥️ Local microphone & speaker pipeline

---

## ✨ Features

* Push-to-talk voice interaction
* Fast CPU-friendly speech recognition
* Persistent conversation memory
* Custom personality via YAML config
* GPT-SoVITS voice cloning support
* Modular architecture (ASR / LLM / TTS)
* Clean, extensible pipeline

---

## 🧠 Architecture

```
Microphone → Whisper → OpenAI → GPT-SoVITS → Speaker
```

Modules:

```
process/
    asr/
    llm/
    tts/
```

---

## 📂 Project Structure

```
project_yukino/
│
├── character_files/
│   └── config.yaml
│
├── process/
│   ├── asr/
│   ├── llm/
│   └── tts/
│
├── audio/
├── main_chat.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

* Python **3.10 or 3.11** (recommended)
* Windows / Linux / macOS
* Microphone + Speaker
* GPT-SoVITS server running locally
* OpenAI API key

CPU works fine. GPU optional.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/tossdace/project_yukino.git
cd project_yukino
```

Create environment:

```bash
python -m venv .venv
```

Activate:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

Edit:

```
character_files/config.yaml
```

Example:

```yaml
OPENAI_API_KEY: "your_key_here"
model: "gpt-4.1-mini"

history_file: "conversation_history.json"

presets:
  default:
    system_prompt: "You are Yukino, a calm and intelligent AI assistant."

sovits_ping_config:
  server_url: "http://127.0.0.1:9880/tts"
  text_lang: "en"
  ref_audio_path: "character_files/main_sample.wav"
  prompt_text: "Hello"
  prompt_lang: "en"
```

---

## 🔊 GPT-SoVITS Server

You **must start the TTS server first**.

Example:

```
http://127.0.0.1:9880
```

If the server is not running → speech generation will fail.

---

## ▶️ Run

```bash
python main_chat.py
```

Controls:

* Press **ENTER** to start recording
* Press **ENTER** again to stop
* Say **exit** to quit

---

## ⚡ Performance Tips

Best Whisper models for CPU:

| Model    | Speed                    |
| -------- | ------------------------ |
| tiny.en  | fastest                  |
| base.en  | good                     |
| small.en | slower but more accurate |

Default uses:

```
base.en + int8
```

---

## 🧩 Modules

| Module | Purpose                         |
| ------ | ------------------------------- |
| ASR    | Audio recording + transcription |
| LLM    | OpenAI conversation + memory    |
| TTS    | GPT-SoVITS speech generation    |

Each component can be replaced independently.

---

## 🛠️ Future Improvements

Planned ideas:

* Wake word detection
* Streaming transcription
* Interruptible speech
* Desktop UI
* Plugin system

---

## 📜 License

MIT License

---

## 👤 Author

**GitHub:** https://github.com/tossdace

---

## ⭐ Acknowledgments

* Faster-Whisper
* OpenAI
* GPT-SoVITS
* FunASR ecosystem

---

## ⚠️ Disclaimer

This project is for research and educational purposes.
Ensure you comply with local laws when cloning voices or recording audio.

dev note

brooooooo iam done
