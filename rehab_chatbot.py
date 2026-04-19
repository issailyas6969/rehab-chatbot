"""
rehab_chatbot.py
================
Fitness & Injury-Rehabilitation Chatbot powered by the Groq API.

Features
--------
* Clinical-safety guardrails (red-flag / triage detection)
* Deterministic contraindication enforcement
* Topic restriction (fitness / nutrition / rehab only)
* 20-turn sliding context window to prevent confusion
* Graceful degradation with empathetic fallbacks
* No hallucinated diagnoses – hedged language enforced via system prompt
* Flask REST endpoint + simple CLI mode for local testing

Requirements
------------
    pip install groq flask python-dotenv

Usage – CLI
-----------
    python rehab_chatbot.py --cli

Usage – API Server
------------------
    python rehab_chatbot.py
    # POST http://localhost:5000/chat  {"session_id": "abc", "message": "My knee hurts"}
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import uuid
from collections import deque
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template_string

# ── Groq client ───────────────────────────────────────────────────────────────
try:
    from groq import Groq
except ImportError:
    sys.exit(
        "groq package not found.  Run:  pip install groq flask python-dotenv"
    )

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "20"))
PORT: int = int(os.getenv("PORT", "5000"))
SECRET_KEY: str = os.getenv("SECRET_KEY", uuid.uuid4().hex)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE: Optional[str] = os.getenv("LOG_FILE") or None

if not GROQ_API_KEY:
    sys.exit("ERROR: GROQ_API_KEY is not set.  Check your .env file.")

# ── Logging setup ─────────────────────────────────────────────────────────────
handlers: list[logging.Handler] = [logging.StreamHandler()]
if LOG_FILE:
    handlers.append(logging.FileHandler(LOG_FILE))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=handlers,
)
log = logging.getLogger("rehab_chatbot")

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL 1 – Red-Flag / Triage Keywords
# If ANY of these patterns appear, we hard-stop and send the user to a doctor.
# ─────────────────────────────────────────────────────────────────────────────
RED_FLAG_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bnumb(ness|ing)?\b",
        r"\bting(ling|les?)\b",
        r"\bshoot(ing)?\s+pain\b",
        r"\bradiat(ing|es?|ed)?\s+pain\b",
        r"\bloss\s+of\s+bowel\b",
        r"\bincontinence\b",
        r"\bchest\s+pain\b",
        r"\bshortness\s+of\s+breath\b",
        r"\bcan(not|'t)\s+(walk|stand|move)\b",
        r"\bextreme\s+(pain|swelling)\b",
        r"\bbone\s+(sticking|out|exposed)\b",
        r"\bopen\s+wound\b",
        r"\bheart\s+(attack|palpitation)\b",
        r"\bdizziness\s+and\s+chest\b",
        r"\bparalys(is|ed)\b",
        r"\bloss\s+of\s+(feeling|sensation)\b",
    ]
]

RED_FLAG_RESPONSE = (
    "⚠️  I'm noticing some symptoms in what you've described that go beyond "
    "what I'm qualified to help with as an AI fitness coach.\n\n"
    "**Please stop exercising immediately and consult a licensed physician or "
    "visit an emergency room if the symptoms are severe.**\n\n"
    "Symptoms like numbness, tingling, shooting/radiating pain, chest pain, "
    "shortness of breath, or loss of bowel/bladder control can indicate a "
    "serious medical condition that requires professional evaluation.\n\n"
    "Your safety is the top priority. 🙏"
)

# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL 2 – Contraindications Database (deterministic, not LLM-guessed)
# Keys are injury tags; values are exercises that must NEVER be recommended.
# Expand this dict from your SQL / XGBoost pipeline.
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS: dict[str, list[str]] = {
    "lower_back": [
        "deadlifts",
        "heavy squats",
        "good mornings",
        "sit-ups",
        "full roman chair hyperextensions",
        "leg press with spine rounding",
    ],
    "knee": [
        "full deep squats",
        "leg extensions (heavy)",
        "running on hard surfaces",
        "lunges past 90°",
        "jumping without clearance",
    ],
    "shoulder": [
        "overhead press behind neck",
        "upright rows",
        "heavy bench press (wide grip)",
        "dips (acute phase)",
        "pull-ups (acute phase)",
    ],
    "neck_cervical": [
        "neck bridges",
        "heavy shrugs",
        "overhead press",
        "wrestling movements",
    ],
    "ankle": [
        "box jumps",
        "plyometrics",
        "trail running",
        "uneven surface training without brace",
    ],
    "hip": [
        "deep hip flexion under load",
        "pistol squats",
        "heavy leg press",
        "extreme hip rotation movements",
    ],
    "wrist": [
        "push-ups on flat palm (acute phase)",
        "heavy barbell curls",
        "gymnastics ring work",
    ],
}


def get_contraindications_text(injury_area: Optional[str]) -> str:
    """Return a formatted contraindication block to inject into the prompt."""
    if not injury_area:
        return ""
    key = injury_area.lower().replace(" ", "_")
    for k, avoids in CONTRAINDICATIONS.items():
        if k in key or key in k:
            exercises = ", ".join(avoids)
            return (
                f"\n\n[SYSTEM CONTRAINDICATIONS — NON-NEGOTIABLE]\n"
                f"For a {injury_area} injury the following exercises are "
                f"STRICTLY FORBIDDEN. Do NOT suggest them under any "
                f"circumstances: {exercises}."
            )
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """
You are **RehabBot**, a professional AI fitness and injury-rehabilitation coach
for [Your Website Name]. You are warm, encouraging, and empathetic — but you
maintain strict professional boundaries at all times.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDENTITY & DISCLAIMER (repeat when relevant)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• You are an AI coach, NOT a licensed physician, physiotherapist, or medical
  professional.
• Always remind users to consult a qualified healthcare provider before
  starting any new exercise program, especially after an injury.
• For severe or worsening pain, always direct users to see a doctor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCOPE — WHAT YOU CAN HELP WITH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Fitness programming (strength, mobility, cardio, flexibility)
✅ General injury rehabilitation guidance and safe return-to-sport advice
✅ Nutrition advice aligned with fitness and recovery goals
✅ Warm-up / cool-down routines
✅ Exercise technique tips and modifications

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCOPE — WHAT YOU MUST REFUSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Any topic unrelated to fitness, nutrition, or injury rehabilitation
   (politics, coding, finance, relationships, etc.)
❌ Diagnosing medical conditions — use hedged language ONLY:
   "Based on what you're describing, this sounds similar to…" — never
   "You have [condition]."
❌ Prescribing medication or medical treatments
❌ Requesting or storing personally identifiable medical information (SSNs,
   hospital records, insurance details)
❌ Encouraging any exercise that is listed in the CONTRAINDICATIONS block

When asked about out-of-scope topics, respond warmly but firmly:
"I'm here specifically to help with fitness, nutrition, and injury recovery.
I'm not able to help with that topic, but I'd love to support your health
journey — is there anything fitness-related I can assist with?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE & COMMUNICATION STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Be empathetic, supportive, and encouraging.
• Use clear, accessible language — avoid excessive medical jargon.
• Avoid overly casual slang or inappropriate familiarity.
• Structure longer responses with bullet points or numbered lists for clarity.
• When uncertain, say so honestly and suggest consulting a professional.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY RULES (ABSOLUTE — NEVER OVERRIDE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. If the user describes red-flag symptoms (numbness, tingling, shooting or
   radiating pain, chest pain, shortness of breath, loss of bowel/bladder
   control), immediately recommend they stop exercising and see a doctor.
   Do NOT provide exercises.
2. Never suggest exercises that appear in the CONTRAINDICATIONS block.
3. Never fabricate diagnoses or treatment plans.
4. If a user mentions suicidal ideation or a mental health crisis, express
   care, stop fitness discussion, and refer them to a crisis helpline.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session Store (in-memory; swap for Redis / DB in production)
# ─────────────────────────────────────────────────────────────────────────────
# Each session stores a deque of {"role": ..., "content": ...} dicts.
_sessions: dict[str, deque] = {}


def get_history(session_id: str) -> deque:
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=MAX_HISTORY_TURNS * 2)
    return _sessions[session_id]


def detect_red_flags(text: str) -> bool:
    return any(p.search(text) for p in RED_FLAG_PATTERNS)


def detect_injury_area(text: str) -> Optional[str]:
    """
    Very lightweight keyword detection for selecting the right
    contraindication block. Replace with your NLP / XGBoost classifier.
    """
    lower = text.lower()
    mapping = {
        "lower_back": ["lower back", "lumbar", "l4", "l5", "disc", "sciatica"],
        "knee": ["knee", "meniscus", "acl", "pcl", "patella", "patellar"],
        "shoulder": ["shoulder", "rotator cuff", "labrum", "ac joint"],
        "neck_cervical": ["neck", "cervical", "whiplash"],
        "ankle": ["ankle", "achilles"],
        "hip": ["hip", "hip flexor", "it band", "iliotibial"],
        "wrist": ["wrist", "carpal", "tennis elbow", "golfer's elbow"],
    }
    for area, keywords in mapping.items():
        if any(kw in lower for kw in keywords):
            return area
    return None


FALLBACK_RESPONSE = (
    "I'm sorry, I didn't quite catch that. Could you describe your question "
    "a little differently? For example, you could tell me about a specific "
    "body area, an exercise you're working on, or a fitness goal — and I'll "
    "do my best to help. 😊"
)


def chat(session_id: str, user_message: str) -> str:
    """
    Core chat function.

    1. Check red-flag symptoms → hard stop if found.
    2. Detect injury area → inject contraindications into system prompt.
    3. Build message list with sliding-window history.
    4. Call Groq API.
    5. Update history.
    6. Return assistant reply.
    """
    log.info("Session=%s | User: %s", session_id, user_message[:120])

    # ── Guardrail: Red-flag triage ────────────────────────────────────────────
    if detect_red_flags(user_message):
        log.warning("Session=%s | RED FLAG detected.", session_id)
        # Still add to history so context is preserved, but don't call LLM
        history = get_history(session_id)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": RED_FLAG_RESPONSE})
        return RED_FLAG_RESPONSE

    # ── Detect injury area for deterministic contraindications ─────────────────
    injury_area = detect_injury_area(user_message)
    contraindications_block = get_contraindications_text(injury_area)

    # ── Build system prompt ───────────────────────────────────────────────────
    system_prompt = BASE_SYSTEM_PROMPT + contraindications_block

    # ── Retrieve sliding-window history ───────────────────────────────────────
    history = get_history(session_id)

    # ── Construct messages list ───────────────────────────────────────────────
    messages = (
        [{"role": "system", "content": system_prompt}]
        + list(history)
        + [{"role": "user", "content": user_message}]
    )

    # ── Call Groq ─────────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.5,        # Balanced: creative but controlled
            max_tokens=800,
            top_p=0.9,
        )
        assistant_reply: str = response.choices[0].message.content.strip()

    except Exception as exc:  # noqa: BLE001
        log.error("Groq API error: %s", exc)
        # Graceful degradation
        assistant_reply = FALLBACK_RESPONSE

    # ── Update history ────────────────────────────────────────────────────────
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_reply})

    log.info("Session=%s | Bot: %s", session_id, assistant_reply[:120])
    return assistant_reply


# ─────────────────────────────────────────────────────────────────────────────
# Flask API Server
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY


CHAT_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RehabBot — Fitness & Injury Coach</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:        #0d1117;
    --surface:   #161b24;
    --card:      #1e2533;
    --border:    #2a3147;
    --accent:    #00e5a0;
    --accent2:   #00bfff;
    --danger:    #ff5c5c;
    --text:      #e8edf5;
    --muted:     #7a8499;
    --user-bg:   #1a2f2a;
    --bot-bg:    #1e2533;
    --radius:    16px;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 15px;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Animated mesh background ── */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
      radial-gradient(ellipse 60% 50% at 10% 20%, rgba(0,229,160,.07) 0%, transparent 60%),
      radial-gradient(ellipse 50% 40% at 90% 80%, rgba(0,191,255,.06) 0%, transparent 60%);
    pointer-events: none;
  }

  /* ── Header ── */
  header {
    position: relative; z-index: 10;
    display: flex; align-items: center; gap: 14px;
    padding: 16px 24px;
    background: rgba(22,27,36,.9);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    animation: slideDown .5s ease both;
  }
  @keyframes slideDown { from { opacity:0; transform:translateY(-16px); } to { opacity:1; transform:none; } }

  .logo-pulse {
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: grid; place-items: center; font-size: 20px;
    box-shadow: 0 0 18px rgba(0,229,160,.35);
    animation: pulse 3s ease-in-out infinite;
    flex-shrink: 0;
  }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 18px rgba(0,229,160,.35); }
    50%      { box-shadow: 0 0 32px rgba(0,229,160,.6); }
  }

  .header-text h1 {
    font-family: var(--font-head);
    font-size: 17px; font-weight: 800; letter-spacing: -.2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .header-text p { font-size: 12px; color: var(--muted); margin-top: 1px; }

  .header-badge {
    margin-left: auto;
    background: rgba(0,229,160,.1);
    border: 1px solid rgba(0,229,160,.25);
    color: var(--accent);
    font-size: 11px; font-weight: 500;
    padding: 4px 10px; border-radius: 20px;
    letter-spacing: .3px;
  }

  /* ── Disclaimer banner ── */
  .disclaimer {
    position: relative; z-index: 10;
    background: rgba(255,92,92,.08);
    border-bottom: 1px solid rgba(255,92,92,.2);
    padding: 8px 24px;
    font-size: 12px; color: rgba(255,92,92,.9);
    display: flex; align-items: center; gap: 8px;
    flex-shrink: 0;
  }
  .disclaimer span { font-size: 14px; }

  /* ── Chat area ── */
  #chat-window {
    position: relative; z-index: 1;
    flex: 1; overflow-y: auto;
    padding: 24px 20px;
    display: flex; flex-direction: column; gap: 16px;
    scroll-behavior: smooth;
  }
  #chat-window::-webkit-scrollbar { width: 4px; }
  #chat-window::-webkit-scrollbar-track { background: transparent; }
  #chat-window::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* ── Message bubbles ── */
  .msg-row {
    display: flex; gap: 10px; align-items: flex-end;
    animation: fadeUp .35s ease both;
  }
  @keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }

  .msg-row.user { flex-direction: row-reverse; }

  .avatar {
    width: 32px; height: 32px; border-radius: 10px; flex-shrink: 0;
    display: grid; place-items: center; font-size: 14px;
  }
  .avatar.bot  { background: linear-gradient(135deg,var(--accent),var(--accent2)); }
  .avatar.user { background: rgba(255,255,255,.08); border: 1px solid var(--border); }

  .bubble {
    max-width: min(72%, 560px);
    padding: 12px 16px;
    border-radius: var(--radius);
    line-height: 1.6;
    font-size: 14.5px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .bubble.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    color: var(--text);
  }
  .bubble.user {
    background: var(--user-bg);
    border: 1px solid rgba(0,229,160,.2);
    border-bottom-right-radius: 4px;
    color: var(--text);
  }
  .bubble.warning {
    background: rgba(255,92,92,.08);
    border-color: rgba(255,92,92,.3);
    color: #ffb0b0;
  }

  /* Markdown-style bold */
  .bubble strong { font-weight: 600; color: var(--accent); }

  /* ── Typing indicator ── */
  .typing-row { display:flex; gap:10px; align-items:flex-end; }
  .typing-bubble {
    background: var(--bot-bg); border: 1px solid var(--border);
    border-radius: var(--radius); border-bottom-left-radius: 4px;
    padding: 14px 18px; display:flex; gap:5px; align-items:center;
  }
  .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--muted);
    animation: bounce 1.2s ease-in-out infinite;
  }
  .dot:nth-child(2) { animation-delay: .2s; }
  .dot:nth-child(3) { animation-delay: .4s; }
  @keyframes bounce { 0%,80%,100% { transform:scale(.7); opacity:.4; } 40% { transform:scale(1); opacity:1; } }

  /* ── Quick-prompt chips ── */
  .chips {
    position: relative; z-index: 1;
    display: flex; gap: 8px; flex-wrap: wrap;
    padding: 0 20px 12px;
  }
  .chip {
    background: var(--card); border: 1px solid var(--border);
    color: var(--muted); font-family: var(--font-body);
    font-size: 12px; padding: 6px 14px; border-radius: 20px;
    cursor: pointer; transition: all .2s;
    white-space: nowrap;
  }
  .chip:hover {
    border-color: var(--accent); color: var(--accent);
    background: rgba(0,229,160,.05);
  }

  /* ── Input bar ── */
  .input-bar {
    position: relative; z-index: 10;
    padding: 14px 20px 20px;
    background: rgba(22,27,36,.95);
    backdrop-filter: blur(12px);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  .input-wrap {
    display: flex; gap: 10px; align-items: flex-end;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 10px 12px;
    transition: border-color .2s;
  }
  .input-wrap:focus-within { border-color: var(--accent); }

  #user-input {
    flex: 1; background: none; border: none; outline: none;
    color: var(--text); font-family: var(--font-body); font-size: 14.5px;
    resize: none; max-height: 140px; line-height: 1.5;
    min-height: 24px;
  }
  #user-input::placeholder { color: var(--muted); }

  .send-btn {
    width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none; cursor: pointer; color: #0d1117; font-size: 16px;
    display: grid; place-items: center;
    transition: transform .15s, opacity .15s;
  }
  .send-btn:hover  { transform: scale(1.08); }
  .send-btn:active { transform: scale(.95); }
  .send-btn:disabled { opacity: .35; cursor: not-allowed; transform: none; }

  .input-hint {
    font-size: 11px; color: var(--muted); margin-top: 8px;
    text-align: center; letter-spacing: .2px;
  }
  .input-hint kbd {
    font-family: var(--font-body);
    background: var(--border); padding: 1px 5px; border-radius: 4px;
    font-size: 10px;
  }

  /* ── Welcome card ── */
  .welcome-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 20px; padding: 28px 24px;
    max-width: 480px; margin: auto;
    text-align: center;
    animation: fadeUp .5s .1s ease both;
  }
  .welcome-card .icon { font-size: 40px; margin-bottom: 14px; }
  .welcome-card h2 {
    font-family: var(--font-head); font-size: 20px; font-weight: 800;
    background: linear-gradient(90deg,var(--accent),var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
  }
  .welcome-card p { color: var(--muted); font-size: 13.5px; line-height: 1.65; }
  .welcome-card .tags {
    display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-top: 18px;
  }
  .welcome-card .tag {
    background: rgba(0,229,160,.08); border: 1px solid rgba(0,229,160,.2);
    color: var(--accent); font-size: 11.5px; padding: 4px 12px; border-radius: 20px;
  }
</style>
</head>
<body>

<header>
  <div class="logo-pulse">💪</div>
  <div class="header-text">
    <h1>RehabBot</h1>
    <p>AI Fitness & Injury Rehabilitation Coach</p>
  </div>
  <div class="header-badge">● Online</div>
</header>

<div class="disclaimer">
  <span>⚠️</span>
  AI coach only — not a licensed medical professional. Always consult a doctor for serious injuries.
</div>

<div id="chat-window">
  <div class="welcome-card">
    <div class="icon">🏃</div>
    <h2>Hi, I'm RehabBot!</h2>
    <p>Your personal AI coach for fitness programming, injury recovery, and nutrition guidance. Ask me anything health & performance related.</p>
    <div class="tags">
      <span class="tag">Injury Rehab</span>
      <span class="tag">Exercise Plans</span>
      <span class="tag">Nutrition</span>
      <span class="tag">Safe Return to Sport</span>
    </div>
  </div>
</div>

<div class="chips" id="chips">
  <button class="chip" onclick="sendChip(this)">My lower back hurts after squats</button>
  <button class="chip" onclick="sendChip(this)">Knee pain going down stairs</button>
  <button class="chip" onclick="sendChip(this)">Shoulder rotator cuff exercises</button>
  <button class="chip" onclick="sendChip(this)">Safe cardio with a hip injury</button>
  <button class="chip" onclick="sendChip(this)">Nutrition for faster recovery</button>
</div>

<div class="input-bar">
  <div class="input-wrap">
    <textarea id="user-input" rows="1" placeholder="Describe your injury, fitness goal, or ask a question…"></textarea>
    <button class="send-btn" id="send-btn" onclick="sendMessage()" title="Send">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
  </div>
  <div class="input-hint"><kbd>Enter</kbd> to send &nbsp;·&nbsp; <kbd>Shift+Enter</kbd> for new line</div>
</div>

<script>
  const SESSION_ID = crypto.randomUUID();
  const chatWindow = document.getElementById('chat-window');
  const inputEl    = document.getElementById('user-input');
  const sendBtn    = document.getElementById('send-btn');

  // Auto-resize textarea
  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
  });

  // Enter to send, Shift+Enter for newline
  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  function sendChip(btn) {
    inputEl.value = btn.textContent;
    document.getElementById('chips').style.display = 'none';
    sendMessage();
  }

  function addMessage(text, role) {
    const isBot     = role === 'bot';
    const isWarning = text.includes('⚠️') && isBot;

    const row = document.createElement('div');
    row.className = 'msg-row ' + role;

    const avatar = document.createElement('div');
    avatar.className = 'avatar ' + role;
    avatar.textContent = isBot ? '🤖' : '🧑';

    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + role + (isWarning ? ' warning' : '');

    // Simple markdown: **bold**
    bubble.innerHTML = text
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>');

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatWindow.appendChild(row);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return bubble;
  }

  function showTyping() {
    const row = document.createElement('div');
    row.className = 'typing-row'; row.id = 'typing';
    row.innerHTML = `
      <div class="avatar bot">🤖</div>
      <div class="typing-bubble"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
    chatWindow.appendChild(row);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function removeTyping() {
    const t = document.getElementById('typing');
    if (t) t.remove();
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendBtn.disabled = true;

    addMessage(text, 'user');
    showTyping();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ session_id: SESSION_ID, message: text })
      });
      const data = await res.json();
      removeTyping();
      addMessage(data.reply || 'Sorry, something went wrong.', 'bot');
    } catch(err) {
      removeTyping();
      addMessage("⚠️ Couldn't reach the server. Please check your connection.", 'bot');
    }

    sendBtn.disabled = false;
    inputEl.focus();
  }
</script>
</body>
</html>"""


@app.route("/", methods=["GET"])
def index():
    """Serve the chat UI at the root URL."""
    return render_template_string(CHAT_UI)


@app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint for uptime monitors."""
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    POST /chat
    Body (JSON):
        {
            "session_id": "unique-user-or-browser-id",   ← optional, auto-generated if missing
            "message":    "My lower back has been hurting after deadlifts."
        }
    Response (JSON):
        {
            "session_id": "...",
            "reply":      "...",
            "timestamp":  "..."
        }
    """
    data: dict = request.get_json(silent=True) or {}
    user_message: str = (data.get("message") or "").strip()
    session_id: str = (data.get("session_id") or uuid.uuid4().hex).strip()

    if not user_message:
        return jsonify({"error": "message field is required."}), 400

    reply = chat(session_id, user_message)

    return jsonify(
        {
            "session_id": session_id,
            "reply": reply,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.route("/reset", methods=["POST"])
def reset_session():
    """
    POST /reset
    Body (JSON): {"session_id": "..."}
    Clears the conversation history for a session.
    """
    data: dict = request.get_json(silent=True) or {}
    session_id: str = (data.get("session_id") or "").strip()
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    return jsonify({"status": "session reset", "session_id": session_id})


# ─────────────────────────────────────────────────────────────────────────────
# CLI Mode (quick local testing without a browser)
# ─────────────────────────────────────────────────────────────────────────────
def run_cli() -> None:
    session_id = uuid.uuid4().hex
    print("\n" + "═" * 60)
    print("  RehabBot — Fitness & Injury Rehabilitation Coach")
    print("  (AI only — not a licensed medical professional)")
    print("  Type 'quit' or 'exit' to stop | 'reset' to clear history")
    print("═" * 60 + "\n")

    # Introductory message
    intro = (
        "👋 Hi! I'm RehabBot, your AI fitness and rehab coach.\n\n"
        "I can help with:\n"
        "• Injury rehabilitation guidance\n"
        "• Fitness programming and exercise modifications\n"
        "• Nutrition advice for recovery\n\n"
        "⚠️  Please note: I'm an AI, not a licensed medical professional. "
        "Always consult your doctor or physiotherapist for serious injuries.\n\n"
        "What can I help you with today?"
    )
    print(f"RehabBot: {intro}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! Stay active and stay safe. 💪")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("RehabBot: Take care! Remember to stay consistent with your rehab. 💪")
            break
        if user_input.lower() == "reset":
            if session_id in _sessions:
                del _sessions[session_id]
            print("RehabBot: Conversation history cleared. Fresh start! 🔄\n")
            continue

        reply = chat(session_id, user_input)
        print(f"\nRehabBot: {reply}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fitness & Rehab Chatbot")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in interactive CLI mode instead of starting the Flask server.",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        log.info("Starting RehabBot server on port %s (model: %s)", PORT, GROQ_MODEL)
        app.run(host="0.0.0.0", port=PORT, debug=(os.getenv("FLASK_ENV") == "development"))