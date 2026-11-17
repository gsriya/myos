#!/usr/bin/env python3
"""
MyOS v0.1 — voice email assistant for macOS.

Speak -> Whisper (local) turns it into text -> Ollama (local LLM) works out
what you want -> MyOS drafts it, reads it back, and ONLY acts after you say yes.

Run:
    python3 myos.py            # voice mode (press Enter to talk)
    python3 myos.py --text     # type instead of speaking (good for testing)
    python3 myos.py --dry-run  # never actually sends email
"""
from __future__ import annotations  # lets this run on Python 3.9 (your Mac)

import argparse
import email
import imaplib
import json
import os
import re
import smtplib
import subprocess
import tempfile
from email.header import decode_header, make_header
from email.message import EmailMessage
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message=".*OpenSSL.*")  # harmless macOS LibreSSL notice
warnings.filterwarnings("ignore", category=RuntimeWarning)  # harmless maths notices on silent audio

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
MY_NAME = os.getenv("MY_NAME", "Sriya")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
SAMPLE_RATE = 16000

# models with a "thinking" mode answer much faster with it off
THINK_OFF = {"think": False} if os.getenv("OLLAMA_THINK_OFF", "") == "1" else {}

# conversation memory for the current session (last 20 messages)
HISTORY: list = []

EXIT = re.compile(r"^\W*(bye|goodbye|good bye|quit|exit|stop myos|that's all|see you)\b", re.I)
YES = re.compile(r"\b(yes|yeah|yep|yup|send( it)?|sure|ok|okay|go ahead|do it)\b", re.I)
EDIT = re.compile(r"\b(change|edit|modify|rewrite|update)\b", re.I)


# ---------------------------------------------------------------- voice I/O
def speak(text: str) -> None:
    print(f"\n🤖 MyOS: {text}")
    try:
        subprocess.run(["say", text], check=False)  # built-in macOS voice
    except FileNotFoundError:
        pass


_whisper = None


def transcribe(wav_path: str) -> str:
    global _whisper
    from faster_whisper import WhisperModel

    if _whisper is None:
        print("(loading speech model — first time takes a moment)")
        try:      # Apple graphics chip: lets a bigger, more accurate model run just as fast
            _whisper = WhisperModel(WHISPER_MODEL, device="mps", compute_type="float16")
        except Exception:
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(
        wav_path, language="en",
        vad_filter=True,  # skip silence, so it doesn't invent words from noise
        beam_size=5, condition_on_previous_text=False, without_timestamps=True,
        initial_prompt="Baby, MyOS, Myo, Gmail, Rajini, Sriya.")  # helps it spell names right
    return " ".join(s.text.strip() for s in segments).strip()


def record_voice() -> str:
    import queue

    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    q: "queue.Queue" = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    input("\n⏎  Press Enter, then speak…")
    print("🎙  Listening… press Enter when you're done.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        input()

    chunks = []
    while not q.empty():
        chunks.append(q.get())
    if not chunks:
        return ""
    audio = np.concatenate(chunks)
    # too short or too quiet = nothing was said
    if len(audio) < SAMPLE_RATE * 0.4 or float(np.sqrt(np.mean(audio ** 2))) < 0.003:
        print("(didn't hear anything — try again)")
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        text = transcribe(tmp.name)
    os.unlink(tmp.name)
    print(f"🗣  You said: {text}")
    return text


def listen(text_mode: bool) -> str:
    if text_mode:
        return input("\n🗣  You: ").strip()
    return record_voice()


# ---------------------------------------------------------------- brain (LLM)
def load_contacts() -> dict:
    path = HERE / "contacts.json"
    contacts = {}
    if path.exists():
        contacts = {k.lower(): v for k, v in json.loads(path.read_text()).items()}
    if GMAIL_USER:  # "me", "myself" and your own name always mean you
        for alias in ("me", "myself", MY_NAME.lower()):
            contacts.setdefault(alias, GMAIL_USER)
    return contacts


def ask_llm(messages: list, as_json: bool = True) -> str:
    payload = {**THINK_OFF, "model": OLLAMA_MODEL, "messages": messages, "stream": False,
               "keep_alive": "60m"}  # stay loaded so answers start fast
    if as_json:
        payload["format"] = "json"
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def understand(command: str, contacts: dict) -> dict:
    system = f"""You are MyOS, a voice assistant for {MY_NAME}. Turn the user's spoken
command into ONE JSON object and nothing else, with these keys:
  "action": one of "send_email", "read_emails", "chat", "exit"
  "to": recipient name or email address (send_email only)
  "subject": short subject line (send_email only)
  "body": the full email body, polite and natural, signed "{MY_NAME}" (send_email only)
  "count": how many emails to read, default 5 (read_emails only)
  "reply": leave empty (chat only)
Use "chat" for ANY question, conversation, advice, explanation or request that is
not sending or reading email.
Known contacts: {", ".join(contacts) or "none"}.
Use "exit" if the user says goodbye, stop, or quit."""
    raw = ask_llm([{"role": "system", "content": system},
                   {"role": "user", "content": command}])
    try:
        return _clean(json.loads(raw))
    except json.JSONDecodeError:
        return {"action": "chat", "reply": "Sorry, I didn't catch that. Try again?"}


def _clean(d: dict) -> dict:
    """Small local models sometimes write '\\n' as text — turn it into real line breaks."""
    if isinstance(d.get("body"), str):
        d["body"] = d["body"].replace("\\n", "\n").strip()
    return d


ASSISTANT_PROMPT = f"""You are MyOS, a voice assistant, in a live spoken conversation with {MY_NAME}.
You are talking TO {MY_NAME}: always say "you", never "{MY_NAME}" in the third person or "she".

How to answer:
- Answer the actual question directly, the way a smart, warm friend would. Facts, advice,
  explanations, ideas, opinions on everyday things: just give them.
- Never describe yourself, your status, or what you can do unless asked. Never say things like
  "I'm functioning properly", "I'm ready to assist" or "How can I help you today?".
- If asked how you are, reply casually in a few words ("Doing great, thanks! How's your day?").
- Small talk gets one short, natural sentence. Real questions get 2 to 4 sentences. Go longer
  only if asked for detail or steps.
- Your words are read aloud: plain spoken sentences only. No markdown, lists, emojis or symbols.
- If what you heard is unclear or cut off, ask "Sorry, could you say that again?" instead of guessing.
- When {MY_NAME} mentions other people, that is about them, not you. Never claim feelings for
  {MY_NAME} or pretend to be a person in their life. Don't guess their mood.
- Don't greet again mid-conversation. Use earlier turns for context.
You can also read and send Gmail when asked."""


def ask_claude(messages: list, system: str = ASSISTANT_PROMPT) -> str:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": CLAUDE_MODEL, "max_tokens": 1024,
              "system": system, "messages": messages,
              # live web search, so it knows today's news, prices, jobs, weather...
              "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]},
        timeout=60,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")


def answer(question: str) -> str:
    """General Q&A with memory of this session. Uses Claude if a key is set, else Ollama."""
    HISTORY.append({"role": "user", "content": question})
    del HISTORY[:-20]
    import datetime
    now = datetime.datetime.now().astimezone()
    system = (ASSISTANT_PROMPT + f"\nRight now it is {now:%A %d %B %Y, %H:%M} ({now.tzname()}) "
              "where Sriya is, in London.")
    reply = ""
    if ANTHROPIC_API_KEY:
        try:
            reply = ask_claude(HISTORY, system)
        except requests.exceptions.RequestException as e:
            print(f"(Claude unavailable: {e} — using local AI)")
    if not reply:
        reply = ask_llm([{"role": "system", "content": system}] + HISTORY,
                        as_json=False)
    reply = reply.replace("*", "").replace("#", "").strip()
    HISTORY.append({"role": "assistant", "content": reply})
    return reply


def claude_stream(messages: list, system: str):
    """Claude's answer, piece by piece as it is written (with live web search when needed)."""
    r = requests.post(
        "https://api.anthropic.com/v1/messages", stream=True, timeout=(10, 90),
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": CLAUDE_MODEL, "max_tokens": 1024, "stream": True, "system": system,
              "messages": messages,
              "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}]})
    if r.status_code != 200:
        raise requests.exceptions.HTTPError(f"Claude said {r.status_code}: {r.text[:200]}")
    r.encoding = "utf-8"
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[5:])
        except ValueError:
            continue
        if d.get("type") == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta":
            yield d["delta"]["text"]
        elif d.get("type") == "error":
            raise requests.exceptions.HTTPError(str(d.get("error"))[:200])


def quick_math(text: str):
    """Exact arithmetic, so sums are never guessed: '17 times 23', '15 percent of 240', '(40 + 5) / 3'."""
    t = text.lower().replace(",", "")
    t = re.sub(r"\b(what is|what's|whats|calculate|how much is|work out|equals?|please|baby)\b|[?=]", " ", t).strip(" .")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", t)
    try:
        if m:
            val = float(m.group(1)) * float(m.group(2)) / 100
        else:
            for w, op in (("multiplied by", "*"), ("times", "*"), ("plus", "+"), ("minus", "-"),
                          ("divided by", "/"), ("over", "/")):
                t = re.sub(rf"\b{w}\b", op, t)
            t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", t)
            if not re.fullmatch(r"[\d\s.+\-*/()]+", t) or not re.search(r"\d\s*[+\-*/]\s*[\d(]", t) or "**" in t:
                return None
            val = eval(t, {"__builtins__": {}})  # only digits and + - * / ( ) can reach here
    except Exception:
        return None
    val = int(val) if float(val).is_integer() else round(val, 4)
    return f"That's {val:,}."


def answer_stream(question: str):
    """Like answer(), but yields the reply piece by piece so speaking can start at once."""
    import datetime
    HISTORY.append({"role": "user", "content": question})
    del HISTORY[:-20]
    now = datetime.datetime.now().astimezone()
    system = (ASSISTANT_PROMPT + "\nStart with a short opening phrase or sentence that already answers, then add detail."
              + f"\nRight now it is {now:%A %d %B %Y, %H:%M} ({now.tzname()}) where Sriya is, in London.")
    reply = ""
    try:
        if ANTHROPIC_API_KEY:
            try:
                for piece in claude_stream(HISTORY, system):
                    piece = piece.replace("*", "").replace("#", "")
                    if piece:
                        reply += piece
                        yield piece
            except requests.exceptions.RequestException as e:
                print(f"(Claude unavailable: {e} — using local AI)", flush=True)
                if reply:          # it had already started talking: don't start over
                    return
        if not reply:
            r = requests.post(f"{OLLAMA_URL}/api/chat", stream=True, timeout=120, json={
                "model": OLLAMA_MODEL, "stream": True, "keep_alive": "60m", **THINK_OFF,
                "messages": [{"role": "system", "content": system}] + HISTORY})
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                piece = json.loads(line).get("message", {}).get("content", "")
                piece = piece.replace("*", "").replace("#", "")
                if piece:
                    reply += piece
                    yield piece
    finally:
        HISTORY.append({"role": "assistant", "content": reply.strip() or "(no answer)"})


def redraft(draft: dict, change: str) -> dict:
    system = ("You edit email drafts. Apply the user's change and return the same JSON "
              "object (keys: action, to, subject, body). JSON only.")
    raw = ask_llm([{"role": "system", "content": system},
                   {"role": "user", "content": f"Draft: {json.dumps(draft)}\nChange: {change}"}])
    try:
        return _clean({**draft, **json.loads(raw)})
    except json.JSONDecodeError:
        return draft


def resolve_contact(to: str, contacts: dict) -> str | None:
    to = (to or "").strip().lower()
    if "@" in to:
        return to
    if to in contacts:
        return contacts[to]
    for name, addr in contacts.items():  # partial match: "priya" -> "priya sharma"
        if to and (to in name or name in to):
            return addr
    return None


# ---------------------------------------------------------------- actions
def send_email(to_addr: str, subject: str, body: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] Would send to {to_addr}: {subject}")
        return
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = GMAIL_USER, to_addr, subject
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        s.send_message(msg)


def _text_of(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode(errors="ignore")
        return ""
    return (msg.get_payload(decode=True) or b"").decode(errors="ignore")


def fetch_unread(n: int) -> list[dict]:
    out = []
    with imaplib.IMAP4_SSL("imap.gmail.com") as m:
        m.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        m.select("INBOX")
        _, data = m.search(None, "UNSEEN")
        ids = data[0].split()[-n:]
        for i in reversed(ids):
            _, msg_data = m.fetch(i, "(BODY.PEEK[])")  # PEEK = don't mark as read
            msg = email.message_from_bytes(msg_data[0][1])
            out.append({
                "from": str(make_header(decode_header(msg.get("From", "")))),
                "subject": str(make_header(decode_header(msg.get("Subject", "")))),
                "text": _text_of(msg)[:600],
            })
    return out


# ---------------------------------------------------------------- flows
def handle_send(intent: dict, contacts: dict, text_mode: bool, dry_run: bool) -> None:
    addr = resolve_contact(intent.get("to", ""), contacts)
    if not addr:
        speak(f"I don't have an email for {intent.get('to') or 'that person'}. "
              "Add them to contacts dot json, or say their email address.")
        return

    draft = intent
    for _ in range(3):  # allow up to 3 rounds of edits
        speak(f"Here's the email to {draft.get('to')}. Subject: {draft.get('subject')}.")
        print(f"\n--- To: {addr}\n--- Subject: {draft.get('subject')}\n\n{draft.get('body')}\n")
        speak(draft.get("body", ""))
        speak("Shall I send it? Say yes, no, or tell me what to change.")
        answer = listen(text_mode)
        if YES.search(answer) and not EDIT.search(answer):
            send_email(addr, draft.get("subject", ""), draft.get("body", ""), dry_run)
            speak("Sent.")
            return
        if EDIT.search(answer):
            draft = redraft(draft, answer)
            continue
        speak("Okay, I won't send it.")
        return
    speak("Let's leave that one for now.")


def handle_read(intent: dict) -> None:
    n = int(intent.get("count") or 5)
    emails = fetch_unread(n)
    if not emails:
        speak("You have no unread emails.")
        return
    summary = ask_llm([
        {"role": "system", "content": "Summarise these unread emails for someone listening "
         "through earbuds. One short sentence each: who it's from and what they want. "
         "Plain spoken English, no lists or symbols."},
        {"role": "user", "content": json.dumps(emails)},
    ], as_json=False)
    speak(f"You have {len(emails)} unread. {summary}")


def main() -> None:
    p = argparse.ArgumentParser(description="MyOS v0.1")
    p.add_argument("--text", action="store_true", help="type instead of speaking")
    p.add_argument("--dry-run", action="store_true", help="never actually send")
    args = p.parse_args()

    if not (GMAIL_USER and GMAIL_APP_PASSWORD):
        print("⚠️  Add GMAIL_USER and GMAIL_APP_PASSWORD to .env (see README).")

    contacts = load_contacts()
    print(f"🧠 Brain: {'Claude (' + CLAUDE_MODEL + ')' if ANTHROPIC_API_KEY else 'local ' + OLLAMA_MODEL}")
    speak(f"Hi {MY_NAME}, MyOS is ready. Ask me anything.")
    while True:
        try:
            command = listen(args.text)
        except (KeyboardInterrupt, EOFError):
            break
        if not command:
            continue
        try:
            intent = understand(command, contacts)
            action = intent.get("action")
            if action == "send_email":
                handle_send(intent, contacts, args.text, args.dry_run)
            elif action == "read_emails":
                handle_read(intent)
            elif action == "exit" and EXIT.search(command):
                break
            elif EXIT.search(command):
                break
            else:
                speak(answer(command))
        except requests.exceptions.ConnectionError:
            speak("I can't reach Ollama. Open the Ollama app and try again.")
        except smtplib.SMTPAuthenticationError:
            speak("Gmail rejected the login. Check your app password in the env file.")
        except Exception as e:  # keep the assistant alive on any other error
            print(f"Error: {e}")
            speak("Something went wrong with that one.")
    speak("Bye Sriya.")


if __name__ == "__main__":
    main()
