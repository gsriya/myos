#!/usr/bin/env python3
"""
MyOS app — runs the MyOS brain behind a friendly web face.

    python server.py            # opens http://localhost:8765
    python server.py --dry-run  # never actually sends email
"""
from __future__ import annotations

import argparse
import atexit
import subprocess
import imaplib
import json
import os
import smtplib
import tempfile
import threading
import webbrowser

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

import myos
import reminders

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
STATE = {"draft": None, "addr": None, "rem": None, "voice": "nat:af_heart"}
FEED: list = []   # things MyOS said on its own (alarms, reminders) for the page to show
DRY_RUN = False
MAILISH = myos.re.compile(r"\b(e-?mails?|mails?|inbox|unread|send|reply|draft|gmail)\b", myos.re.I)
NO = myos.re.compile(r"\b(no|nope|cancel|stop|don'?t|never ?mind)\b", myos.re.I)


def gmail_ready() -> bool:
    return bool(myos.GMAIL_USER and myos.GMAIL_APP_PASSWORD
                and "xxxx" not in myos.GMAIL_APP_PASSWORD)


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    brain = f"Claude · {myos.CLAUDE_MODEL}" if myos.ANTHROPIC_API_KEY else f"Local · {myos.OLLAMA_MODEL}"
    return jsonify(brain=brain, claude=bool(myos.ANTHROPIC_API_KEY), gmail=gmail_ready(), dry_run=DRY_RUN, name=myos.MY_NAME,
                   email=myos.GMAIL_USER)


@app.post("/api/transcribe")
def transcribe():
    f = request.files.get("audio")
    if not f:
        return jsonify(text="")
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        f.save(tmp.name)
    try:
        text = myos.transcribe(tmp.name)
        print(f"🗣  heard: {text!r}", flush=True)
    except Exception as e:
        print("transcribe error:", e)
        text = ""
    finally:
        os.unlink(tmp.name)
    return jsonify(text=text)


@app.post("/api/message")
def message():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return jsonify(type="chat", reply="")
    try:
        if STATE["draft"]:
            return jsonify(**pending_draft(text))
        if STATE["rem"] or reminders.TRIGGER.search(text):
            reply, STATE["rem"] = reminders.handle(text, STATE["rem"])
            return jsonify(type="chat", reply=reply, expects_reply=bool(STATE["rem"]))
        exact = myos.quick_math(text)
        if exact:
            myos.HISTORY += [{"role": "user", "content": text}, {"role": "assistant", "content": exact}]
            return jsonify(type="chat", reply=exact)
        if not MAILISH.search(text):      # ordinary question: answer straight away, streamed
            return stream_answer(text)
        intent = myos.understand(text, myos.load_contacts())
        action = intent.get("action")
        if action == "send_email":
            return jsonify(**start_draft(intent))
        if action == "read_emails":
            return jsonify(**read_emails(intent))
        return stream_answer(text)
    except requests.exceptions.ConnectionError:
        return jsonify(type="chat", reply="I can't reach my brain. Please open the Ollama app and try again.")
    except Exception as e:
        print("error:", e)
        return jsonify(type="chat", reply="Oops, something went wrong with that one. Try again?")


def save_env(updates: dict) -> None:
    path = os.path.join(HERE, ".env")
    lines = open(path).read().splitlines() if os.path.exists(path) else []
    seen = set()
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in updates:
            lines[i] = f"{key}={updates[key]}"
            seen.add(key)
    lines += [f"{k}={v}" for k, v in updates.items() if k not in seen]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o600)  # only you can read it


@app.post("/api/gmail/connect")
def gmail_connect():
    d = request.get_json(silent=True) or {}
    user = (d.get("email") or "").strip()
    pw = "".join((d.get("password") or "").split())  # remove spaces
    if "@" not in user:
        return jsonify(ok=False, error="Please enter your Gmail address.")
    if len(pw) != 16:
        return jsonify(ok=False, error="An app password is exactly 16 letters. Copy it again from Google.")
    try:
        with imaplib.IMAP4_SSL("imap.gmail.com") as m:
            m.login(user, pw)
    except imaplib.IMAP4.error:
        return jsonify(ok=False, error="Google didn't accept that. Check the email and create a fresh app password.")
    except OSError:
        return jsonify(ok=False, error="I couldn't reach Gmail. Check your internet and try again.")
    save_env({"GMAIL_USER": user, "GMAIL_APP_PASSWORD": pw})
    myos.GMAIL_USER, myos.GMAIL_APP_PASSWORD = user, pw
    return jsonify(ok=True)


@app.post("/api/claude/connect")
def claude_connect():
    key = "".join(((request.get_json(silent=True) or {}).get("key") or "").split())
    if not key.startswith("sk-ant-") or len(key) < 40:
        return jsonify(ok=False, error="That doesn't look like a Claude API key. It starts with sk-ant-")
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=30,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": myos.CLAUDE_MODEL, "max_tokens": 5,
                                "messages": [{"role": "user", "content": "hi"}]})
    except requests.exceptions.RequestException:
        return jsonify(ok=False, error="I couldn't reach Claude. Check your internet and try again.")
    if r.status_code == 401:
        return jsonify(ok=False, error="Claude didn't accept that key. Copy it again from the Console.")
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = f"error {r.status_code}"
        return jsonify(ok=False, error="Claude said: " + msg[:160])
    save_env({"ANTHROPIC_API_KEY": key})
    myos.ANTHROPIC_API_KEY = key
    return jsonify(ok=True)


@app.post("/api/claude/disconnect")
def claude_disconnect():
    save_env({"ANTHROPIC_API_KEY": ""})
    myos.ANTHROPIC_API_KEY = ""
    return jsonify(ok=True)


@app.post("/api/practice")
def practice():
    global DRY_RUN
    DRY_RUN = bool((request.get_json(silent=True) or {}).get("on"))
    return jsonify(dry_run=DRY_RUN)


# ------------------------------------------------------------------ your own voice
VOICE_URL = "http://127.0.0.1:8766"
VOICE_PY = os.path.join(HERE, ".voice-venv", "bin", "python")
VOICE_DIR = os.path.join(HERE, "voice")
_voice_proc = None


def start_voice_service() -> None:
    global _voice_proc
    if not os.path.exists(VOICE_PY):
        return
    try:
        requests.get(VOICE_URL + "/health", timeout=1)
        return  # already running
    except requests.exceptions.RequestException:
        pass
    log = open(os.path.join(HERE, "voice.log"), "a")
    _voice_proc = subprocess.Popen([VOICE_PY, os.path.join(HERE, "voice_server.py")],
                                   stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
    atexit.register(lambda: _voice_proc and _voice_proc.terminate())
    print("🎙  starting your-voice service…")


@app.get("/api/voice/status")
def voice_status():
    installed = os.path.exists(VOICE_PY)
    try:
        h = requests.get(VOICE_URL + "/health", timeout=1).json()
    except requests.exceptions.RequestException:
        h = {"ready": False, "natural": False, "has_voice": os.path.exists(os.path.join(VOICE_DIR, "ref.wav"))}
    return jsonify(installed=installed, **h)


def _to_wav24k(src: str, dst: str) -> float:
    """Convert any recording to clean 24 kHz mono WAV, trimmed, max 12 s."""
    import av
    import numpy as np
    import soundfile as sf

    c = av.open(src)
    rs = av.AudioResampler(format="s16", layout="mono", rate=24000)
    parts = []
    for frame in c.decode(c.streams.audio[0]):
        parts += [f.to_ndarray().reshape(-1) for f in rs.resample(frame)]
    parts += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
    c.close()
    audio = np.concatenate(parts).astype(np.float32) / 32768.0
    # trim quiet start/end (keep a little breathing room)
    win = 480
    loud = np.where(np.abs(audio) > max(0.02, np.abs(audio).max() * 0.08))[0]
    if loud.size:
        audio = audio[max(0, loud[0] - win * 5): min(len(audio), loud[-1] + win * 10)]
    audio = audio[: 24000 * 12]
    audio = np.concatenate([audio, np.zeros(int(24000 * 0.25), dtype=np.float32)])  # short pause at end
    sf.write(dst, audio, 24000, subtype="PCM_16")
    return len(audio) / 24000


@app.post("/api/voice/sample")
def voice_sample():
    f = request.files.get("audio")
    if not f:
        return jsonify(ok=False, error="No recording received.")
    os.makedirs(VOICE_DIR, exist_ok=True)
    raw = os.path.join(VOICE_DIR, "raw_upload")
    f.save(raw)
    tmp = os.path.join(VOICE_DIR, "ref_new.wav")
    try:
        seconds = _to_wav24k(raw, tmp)
    except Exception as e:
        print("voice sample error:", e)
        return jsonify(ok=False, error="I couldn't read that recording. Please try again.")
    finally:
        os.path.exists(raw) and os.unlink(raw)
    if seconds < 4:
        os.unlink(tmp)
        return jsonify(ok=False, error="That was too short. Please read the whole sentence (about 10 seconds).")
    text = myos.transcribe(tmp).strip()
    if len(text.split()) < 6:
        os.unlink(tmp)
        return jsonify(ok=False, error="I couldn't hear the words clearly. Try again a bit closer to the mic, somewhere quiet.")
    with open(os.path.join(VOICE_DIR, "ref.txt"), "w") as fh:
        fh.write(text)
    os.replace(tmp, os.path.join(VOICE_DIR, "ref.wav"))  # swap in last, so it is never half-written
    start_voice_service()
    return jsonify(ok=True, text=text, seconds=round(seconds, 1))


@app.post("/api/tts")
def tts():
    d = request.get_json(silent=True) or {}
    text = (d.get("text") or "").strip()
    try:
        r = requests.post(VOICE_URL + "/tts", json={"text": text, "voice": d.get("voice") or "mine"}, timeout=90)
    except requests.exceptions.RequestException:
        start_voice_service()
        return jsonify(error="voice service starting"), 503
    if r.status_code != 200:
        return jsonify(error=r.text[:200]), 503
    return Response(r.content, mimetype="audio/wav")


# ------------------------------------------------------------------ alarms & reminders speak by themselves
def speak_aloud(text: str) -> None:
    """Say something through the Mac's speakers, in the chosen MyOS voice when possible."""
    try:
        v = STATE["voice"]
        if not v:
            raise RuntimeError("mac voice")
        r = requests.post(VOICE_URL + "/tts", json={"text": text, "voice": v}, timeout=60)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(r.content)
        subprocess.run(["afplay", tmp.name], check=False)
        os.unlink(tmp.name)
    except Exception:
        subprocess.run(["say", text], check=False)


def on_reminder(text: str, kind: str) -> None:
    print(f"⏰ {text}", flush=True)
    FEED.append({"id": len(FEED) + 1, "text": text, "kind": kind})
    subprocess.run(["osascript", "-e", "on run argv", "-e",
                    'display notification (item 1 of argv) with title "MyOS" sound name "Glass"',
                    "-e", "end run", text], check=False, capture_output=True)

    def ring():
        for _ in range(3 if kind != "reminder" else 1):
            subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False)
        speak_aloud(text)
    threading.Thread(target=ring, daemon=True).start()


@app.get("/api/feed")
def feed():
    after = int(request.args.get("after", "-1"))
    if after < 0:
        return jsonify(last=len(FEED), items=[])
    return jsonify(last=len(FEED), items=FEED[after:])


@app.post("/api/voice/pref")
def voice_pref():
    STATE["voice"] = (request.get_json(silent=True) or {}).get("voice") or ""
    return jsonify(ok=True)


def stream_answer(text: str) -> Response:
    def gen():
        try:
            for piece in myos.answer_stream(text):
                yield piece
        except requests.exceptions.ConnectionError:
            yield "I can't reach my brain. Please open the Ollama app and try again."
        except Exception as e:
            print("stream error:", e, flush=True)
            yield " Sorry, something went wrong with that one."
    return Response(gen(), mimetype="text/plain; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/reset")
def reset():
    myos.HISTORY.clear()
    STATE.update(draft=None, addr=None, rem=None)
    return jsonify(ok=True)


# ------------------------------------------------------------------ email flows
def start_draft(intent: dict) -> dict:
    addr = myos.resolve_contact(intent.get("to", ""), myos.load_contacts())
    if not addr:
        who = intent.get("to") or "that person"
        return dict(type="chat", reply=f"I don't have an email for {who} yet. "
                    "Add them to contacts.json, or tell me their email address.")
    STATE.update(draft=intent, addr=addr)
    return draft_payload(f"Here's your email to {str(intent.get('to') or '').title()}. Shall I send it?")


def draft_payload(reply: str) -> dict:
    d = STATE["draft"]
    return dict(type="draft", reply=reply, to=str(d.get("to") or "").title(), addr=STATE["addr"],
                subject=d.get("subject", ""), body=d.get("body", ""))


def pending_draft(text: str) -> dict:
    d, addr = STATE["draft"], STATE["addr"]
    if myos.YES.search(text) and not myos.EDIT.search(text) and not NO.search(text):
        STATE.update(draft=None, addr=None)
        if not DRY_RUN and not gmail_ready():
            return dict(type="chat", reply="I need Gmail connected before I can send. Click Connect Gmail at the top right of my window. "
                        "Nothing was sent.")
        try:
            myos.send_email(addr, d.get("subject", ""), d.get("body", ""), DRY_RUN)
        except smtplib.SMTPAuthenticationError:
            return dict(type="chat", reply="Gmail rejected the login. Please check the app password in your .env file.")
        return dict(type="sent", reply="Sent!" + (" (practice mode, nothing really went out)" if DRY_RUN else ""))
    if NO.search(text) and not myos.EDIT.search(text):
        STATE.update(draft=None, addr=None)
        return dict(type="chat", reply="Okay, I won't send it.")
    STATE["draft"] = myos.redraft(d, text)
    return draft_payload("I've updated it. Shall I send it now?")


def read_emails(intent: dict) -> dict:
    if not gmail_ready():
        return dict(type="chat", reply="I need Gmail connected first. Click Connect Gmail at the top right of my window.")
    emails = myos.fetch_unread(int(intent.get("count") or 5))
    if not emails:
        return dict(type="chat", reply="You have no unread emails. Inbox zero!")
    summary = myos.ask_llm([
        {"role": "system", "content": "Summarise these unread emails for someone listening. One short "
         "sentence each: who it's from and what they want. Plain spoken English, no lists or symbols."},
        {"role": "user", "content": json.dumps(emails)},
    ], as_json=False)
    return dict(type="chat", reply=f"You have {len(emails)} unread. {summary}")


# ------------------------------------------------------------------ run
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    DRY_RUN = args.dry_run
    url = f"http://localhost:{args.port}"
    print(f"\n✨ MyOS is running at {url}  (Ctrl+C to stop)\n")
    start_voice_service()
    reminders.init(myos.MY_NAME, on_reminder)

    def warm_up():  # load the ears and the brain now, so the first question isn't slow
        try:
            import numpy as np, soundfile as sf
            tmp = os.path.join(tempfile.gettempdir(), "myos_warm.wav")
            sf.write(tmp, np.zeros(16000, dtype="float32"), 16000)
            myos.transcribe(tmp)
            myos.ask_llm([{"role": "user", "content": "hi"}], as_json=False)
            print("✅ warmed up", flush=True)
        except Exception as e:
            print("warm-up skipped:", e, flush=True)
    threading.Thread(target=warm_up, daemon=True).start()
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(port=args.port, debug=False)
