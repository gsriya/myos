"""
MyOS alarms, timers and reminders.

Understands things like:
  "set an alarm at 6 pm"            "remind me to call mum in 20 minutes"
  "remind me to put my clothes in the washing machine"  -> asks "What time?"
  "set a timer for 10 minutes"      "what reminders do I have"   "cancel my reminders"

Everything is stored in reminders.json next to this file, and checked every 2 seconds.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "reminders.json")
_lock = threading.Lock()
_cfg = {"name": "there", "on_fire": lambda text, kind: None}

TRIGGER = re.compile(r"\b(remind|reminders?|alarms?|timers?|wake me)\b", re.I)
CANCEL = re.compile(r"\b(cancel|delete|clear|remove|stop|never ?mind|forget it)\b", re.I)
LIST = re.compile(r"\b(what|which|list|show|any|do i have|tell me)\b", re.I)

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
         "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90}
UNIT = r"(seconds?|secs?|minutes?|mins?|hours?|hrs?)"
REL = re.compile(rf"\b(?:in|for|after)\s+(\d+|an?|half an?|a couple of|a few)\s*{UNIT}\b", re.I)
CLOCK = re.compile(r"\b(at|by|for|around|about)?\s*(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?\b", re.I)
DAYWORDS = re.compile(r"\b(tomorrow|tonight|today|this (?:morning|afternoon|evening)|in the (?:morning|afternoon|evening)|"
                      r"(?:tomorrow )?(?:morning|afternoon|evening|night))\b", re.I)


def init(name: str, on_fire) -> None:
    _cfg.update(name=name, on_fire=on_fire)
    threading.Thread(target=_loop, daemon=True).start()


# ------------------------------------------------------------------ storage
def _load() -> list:
    try:
        return json.load(open(FILE))
    except (OSError, ValueError):
        return []


def _save(items: list) -> None:
    tmp = FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, indent=1)
    os.replace(tmp, FILE)


def add(kind: str, when: dt.datetime, task: str = "") -> dict:
    item = {"id": uuid.uuid4().hex[:8], "kind": kind, "when": when.isoformat(timespec="seconds"), "task": task}
    with _lock:
        items = _load()
        items.append(item)
        _save(items)
    return item


def _loop() -> None:
    while True:
        time.sleep(2)
        try:
            now = dt.datetime.now()
            with _lock:
                items = _load()
                due = [i for i in items if dt.datetime.fromisoformat(i["when"]) <= now]
                if due:
                    _save([i for i in items if i not in due])
            for i in due:
                _cfg["on_fire"](fire_text(i), i["kind"])
        except Exception as e:  # never let the clock stop
            print("reminder loop error:", e, flush=True)


# ------------------------------------------------------------------ wording
def nice_time(when: dt.datetime, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now()
    t = when.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")
    days = (when.date() - now.date()).days
    return t if days == 0 else f"{t} tomorrow" if days == 1 else f"{t} on {when.strftime('%A')}"


def fire_text(i: dict) -> str:
    name = _cfg["name"]
    if i["kind"] == "alarm":
        return f"{name}, it's {nice_time(dt.datetime.fromisoformat(i['when']))}. This is your alarm."
    if i["kind"] == "timer":
        return f"{name}, your timer is done."
    return f"{name}, it's time to {i['task']}."


def to_you(task: str) -> str:
    """'put my clothes away' -> 'put your clothes away'"""
    swaps = [(r"\bmy\b", "your"), (r"\bmine\b", "yours"), (r"\bmyself\b", "yourself"),
             (r"\bi'm\b", "you're"), (r"\bi am\b", "you are"), (r"\bi\b", "you"), (r"\bme\b", "you")]
    for a, b in swaps:
        task = re.sub(a, b, task, flags=re.I)
    return task.strip(" ,.-")


# ------------------------------------------------------------------ understanding
def _norm(text: str) -> str:
    t = text.lower().replace("p.m.", "pm").replace("a.m.", "am").replace("p.m", "pm").replace("a.m", "am")
    t = t.replace("o'clock", "").replace("half past ", "30 past ")
    for w, n in WORDS.items():
        t = re.sub(rf"\b{w}\b", str(n), t)
    return t


def parse_time(text: str, now: dt.datetime | None = None, loose: bool = False) -> dt.datetime | None:
    """loose=True also accepts a bare number ("6"), for answers to "What time?"."""
    now = now or dt.datetime.now()
    t = _norm(text)

    m = REL.search(t)
    if m:
        q, unit = m.group(1), m.group(2)
        n = {"a": 1, "an": 1, "half a": 0.5, "half an": 0.5, "a couple of": 2, "a few": 3}.get(q) or float(q)
        secs = n * (1 if unit.startswith("s") else 60 if unit.startswith("m") else 3600)
        return now + dt.timedelta(seconds=secs)

    tomorrow = "tomorrow" in t
    hint = "pm" if re.search(r"\b(tonight|evening|afternoon|night)\b", t) else "am" if "morning" in t else None
    base = now + dt.timedelta(days=1) if tomorrow else now

    if "noon" in t or "midday" in t:
        h, mi, ap = 12, 0, "pm"
    elif "midnight" in t:
        h, mi, ap = 0, 0, "24"
    else:
        best = None
        for m in CLOCK.finditer(t):
            prep, hh, mm, ap = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            if not (ap or mm or prep in ("at", "by", "around", "about") or loose):
                continue
            if hh > 23 or (mm and int(mm) > 59):
                continue
            best = (hh, int(mm or 0), ap)
            if ap or mm:
                break
        if not best:
            return None
        h, mi, ap = best

    def at(hour: int) -> dt.datetime:
        return base.replace(hour=hour % 24, minute=mi, second=0, microsecond=0)

    ap = ap or (hint if 1 <= h <= 12 else None)
    if ap == "24" or h > 12 or h == 0:
        when = at(h)
    elif ap:
        when = at(h % 12 + (12 if ap == "pm" else 0))
    else:  # "at 6" -> whichever 6 o'clock comes next
        options = [at(h % 12), at(h % 12 + 12)]
        future = [o for o in options if o > now]
        when = min(future) if future else options[0]
    if when <= now:
        when += dt.timedelta(days=1)
    return when


def extract_task(text: str) -> str:
    t = _norm(text.strip().rstrip(".!?"))
    t = re.sub(r"^.*?\b(?:remind(?:er)?|reminders)\b", "", t)
    t = REL.sub(" ", t)
    t = re.sub(r"\b(?:at|by|around|about|for)\s+\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?\b", " ", t)
    t = re.sub(r"\b\d{1,2}(?:[:.]\d{2})\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", " ", t)
    t = re.sub(r"\b(?:at\s+)?(?:noon|midday|midnight)\b", " ", t)
    t = DAYWORDS.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.-")
    t = re.sub(r"^(?:(?:me|to|about|that|for|please|i need to|i have to|i should)\b[\s,]*)+", "", t)
    t = re.sub(r"(?:[\s,]+(?:at|by|in|on|for|please))+$", "", t)
    return to_you(t)


# ------------------------------------------------------------------ conversation
def handle(text: str, pending: dict | None) -> tuple[str, dict | None]:
    """Returns (what MyOS says, what it is still waiting for or None)."""
    now = dt.datetime.now()

    if pending:
        if CANCEL.search(text):
            return "Okay, cancelled.", None
        if pending["need"] == "task":
            task = extract_task("remind me to " + text) or to_you(text)
            when = parse_time(text, now)
            if not when:
                return "Sure. What time should I remind you?", {"need": "time", "kind": "reminder", "task": task}
            return _confirm("reminder", when, task), None
        when = parse_time(text, now, loose=True)
        if not when:
            if pending.get("tries"):
                return "Sorry, I still didn't get the time. Let's try again later.", None
            return "Sorry, I didn't catch the time. Say something like 6 pm, or in 20 minutes.", {**pending, "tries": 1}
        return _confirm(pending["kind"], when, pending.get("task", "")), None

    low = text.lower()
    if CANCEL.search(low):
        with _lock:
            n = len(_load())
            _save([])
        return ("Done. I've cancelled it." if n == 1 else f"Done. I've cancelled all {n}." if n
                else "You don't have any alarms or reminders set."), None
    if LIST.search(low) and not re.search(r"\bremind me\b", low):
        with _lock:
            items = sorted(_load(), key=lambda i: i["when"])
        if not items:
            return "You don't have any alarms or reminders set.", None
        parts = []
        for i in items[:6]:
            w = nice_time(dt.datetime.fromisoformat(i["when"]), now)
            parts.append(f"{'an alarm' if i['kind'] == 'alarm' else 'a timer' if i['kind'] == 'timer' else 'a reminder to ' + i['task']} at {w}")
        return "You have " + ", and ".join(parts) + ".", None

    kind = "timer" if "timer" in low else "alarm" if re.search(r"\balarm|wake me\b", low) else "reminder"
    when = parse_time(text, now)
    if kind == "reminder":
        task = extract_task(text)
        if not task:
            return "Sure. What should I remind you about?", {"need": "task"}
        if not when:
            return "Sure. What time should I remind you?", {"need": "time", "kind": kind, "task": task}
        return _confirm(kind, when, task), None
    if not when:
        q = "How long should the timer be?" if kind == "timer" else "What time should I set the alarm for?"
        return q, {"need": "time", "kind": kind}
    return _confirm(kind, when, ""), None


def _confirm(kind: str, when: dt.datetime, task: str) -> str:
    add(kind, when, task)
    w = nice_time(when)
    if kind == "alarm":
        return f"Done. Your alarm is set for {w}."
    if kind == "timer":
        mins = max(1, round((when - dt.datetime.now()).total_seconds() / 60))
        return f"Timer set for {mins} minute{'s' if mins != 1 else ''}."
    return f"Okay, I'll remind you to {task} at {w}."
