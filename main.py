# main.py
import os
import re
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

# Load .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
UNLOCK_KEY = os.getenv("UNLOCK_KEY", "dev-unlock-key")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

app = FastAPI(title="AI Humanizer (Standalone)")

# Allow local frontend (Next.js) to call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Models ----
class TextRequest(BaseModel):
    text: str
    style: Optional[str] = "natural"

class DetectResponse(BaseModel):
    human_score: float
    label: str

class RewriteResponse(BaseModel):
    rewritten_text: str
    can_copy: bool

# ---- Helpers ----
WORD_REGEX = re.compile(r"\w+")

def count_words(text: str) -> int:
    return len(WORD_REGEX.findall(text))

def build_system_prompt(style: str) -> str:
    prompts = {
        "natural": "Rewrite the text so it reads naturally and clearly, as if written by a human. Keep meaning intact.",
        "casual": "Rewrite in a friendly, casual tone, like you're talking to a friend.",
        "professional": "Rewrite in a concise, professional tone suitable for work communications.",
        "creative": "Rewrite with creative flair, keeping ideas intact but making it engaging.",
        "academic": "Rewrite in a formal, academic tone suitable for essays."
    }
    return prompts.get(style.lower(), prompts["natural"])

# ---- Endpoints ----

@app.post("/detect", response_model=DetectResponse)
async def detect_text(req: TextRequest):
    """
    Run a light detect-style check using OpenAI by asking model whether text seems human or AI.
    (Not a certified detector — just a heuristic prompt.)
    """
    try:
        prompt = (
            "Rate how likely the following text was written by a human rather than an AI, "
            "on a scale from 0 (definitely AI) to 100 (definitely human). "
            "Respond with a single number 0-100 and a short label 'AI' or 'HUMAN'.\n\n"
            f"Text:\n{req.text}\n\nAnswer format: <score>|<label>\n"
        )

        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20,
        )

        out = resp.choices[0].message.content.strip()
        # parse "87|HUMAN" or "12 | AI"
        if "|" in out:
            score_str, label = out.split("|", 1)
            score = float(re.findall(r"[-+]?\d*\.\d+|\d+", score_str)[0])
            label = label.strip().upper()
        else:
            # fallback: try to extract a number and infer label
            num = re.findall(r"[-+]?\d*\.\d+|\d+", out)
            score = float(num[0]) if num else 50.0
            label = "HUMAN" if score >= 50 else "AI"

        # clamp
        score = max(0.0, min(100.0, score))
        return {"human_score": round(score, 2), "label": label}
    except Exception as e:
        # If OpenAI fails, return a neutral fallback
        return {"human_score": 50.0, "label": "UNKNOWN"}

@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite_text(req: TextRequest, x_unlock_key: Optional[str] = Header(None)):
    """
    Rewrite text using OpenAI. Always returns a preview (rewritten_text).
    can_copy is True only if caller provides X-UNLOCK-KEY header matching UNLOCK_KEY.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="Text is required")

    incoming_words = count_words(req.text)

    system_prompt = build_system_prompt(req.style)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.text}
            ],
            temperature=0.8,
            max_tokens=1024,
        )
        rewritten = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

    # can_copy only if unlock header matches server unlock key (simple dev paywall)
    can_copy = (x_unlock_key is not None and x_unlock_key == UNLOCK_KEY)

    return {"rewritten_text": rewritten, "can_copy": can_copy}
