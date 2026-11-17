# MyOS: a voice assistant that does things for you

MyOS is a hands-free assistant for your Mac. You talk to it and it **does the task**: reads and sends email, sets alarms and reminders, and answers questions. It runs locally by default, so your voice and email stay on your machine.

> Say *"Send a mail to Alex saying I'll be late tomorrow"*. MyOS drafts it, reads it back, and sends it **only after you say yes**.
> Say *"Read my unread emails"* and it summarises them out loud.
> Say *"Remind me to call Mum in 20 minutes"* or *"Set an alarm at 6 pm"*.

## What it can do (v0.2)

- **Voice in:** speech-to-text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), running locally.
- **Brain:** a local LLM through [Ollama](https://ollama.com) (default `llama3.2`). You can optionally add a Claude API key for stronger, streamed answers.
- **Email (Gmail):** read unread mail, draft replies, edit a draft by voice ("make it shorter"), and send only after a spoken "yes". Contacts are matched by name.
- **Alarms, timers and reminders:** natural language ("remind me to…"). If you don't give a time, it asks "What time?". Reminders are stored locally and checked every 2 seconds.
- **Web app:** `server.py` opens a friendly web interface on `http://localhost:8765` with chat, a microphone button, live streaming answers and settings to connect Gmail and Claude.
- **Natural voice (optional, Apple Silicon):** `voice_server.py` speaks replies with Kokoro voices, or in **your own cloned voice** from a short recording (F5-TTS on MLX). Everything stays on the device.
- **Safety:** `--dry-run` never sends anything, and every email needs your spoken confirmation.

## Quick start

```bash
git clone https://github.com/gsriya/myos.git && cd myos
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2                      # the local brain (install the Ollama app first)
cp .env.example .env                      # add your Gmail address + app password
cp contacts.example.json contacts.json    # "name": "email"
```

Gmail app password: turn on 2-Step Verification, go to myaccount.google.com/apppasswords, create one called "MyOS" and paste the 16 letters into `.env`.

### Run

```bash
python3 server.py --dry-run    # web app, nothing really sends
python3 server.py              # web app, live

python3 myos.py --text --dry-run   # terminal: type commands
python3 myos.py                    # terminal: talk (allow the mic)
```

### Optional: natural or cloned voice (Apple Silicon)

```bash
python3.11 -m venv .voice-venv && .voice-venv/bin/pip install -r requirements-voice.txt
```

Put the Kokoro model files (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) in `voice/models/`. To use your own voice, record a short clip in the web app. `server.py` starts the voice service automatically.

## Project structure

```
myos.py            core: listen, understand the request, act (email, answers)
reminders.py       alarms, timers and reminders
server.py          Flask web app + API (port 8765)
voice_server.py    local text-to-speech service (port 8766)
static/index.html  the web interface
```

## Privacy

- `.env`, `contacts.json`, `reminders.json`, voice recordings and model files are **git-ignored** and never leave your machine.
- Speech recognition and the default LLM run locally. Only if you add a Claude API key are questions sent to Anthropic's API.

## Roadmap

- Wake word ("Hey MyOS") so it's fully hands-free
- Outlook and Teams via Microsoft Graph
- WhatsApp and Instagram actions, opening apps and files by voice
- Menu-bar Mac app, then a downloadable release on getmyos.com, free for the first year

Built by Sriya Gurrawar.
