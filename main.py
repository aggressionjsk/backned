import os
import jwt
import httpx
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
CLERK_ISSUER = os.getenv("CLERK_ISSUER")
CLERK_AUDIENCE = os.getenv("CLERK_AUDIENCE")

if not OPENAI_API_KEY or not CLERK_SECRET_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY or CLERK_SECRET_KEY in .env file")

app = FastAPI()

# ---------------------------
# CORS (allow frontend calls)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend local dev URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# OpenAI Client
# ---------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Clerk Auth Helper
# ---------------------------
async def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(
            token,
            CLERK_SECRET_KEY,
            algorithms=["HS256"],
            audience=CLERK_AUDIENCE,
            issuer=CLERK_ISSUER,
        )
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Clerk token: {str(e)}")

# ---------------------------
# Request & Response Models
# ---------------------------
class RewriteRequest(BaseModel):
    text: str
    style: str = "natural"

# ---------------------------
# Rewrite Endpoint
# ---------------------------
@app.post("/rewrite")
async def rewrite_text(req: RewriteRequest, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user ID in token")

    # Fetch user metadata from Clerk
    async with httpx.AsyncClient() as http:
        resp = await http.get(
            f"https://api.clerk.com/v1/users/{user_id}",
            headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"}
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch user metadata")
        user_data = resp.json()

    metadata = user_data.get("public_metadata", {})
    premium = metadata.get("premium", False)

    # Call OpenAI to rewrite
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rewrite the given text to sound natural, human-written, and high quality."},
            {"role": "user", "content": req.text}
        ],
    )
    rewritten = response.choices[0].message.content.strip()

    # Return preview text + copy lock
    return {
        "rewritten_text": rewritten,
        "can_copy": premium  # false until user pays
    }
    
# ---------------------------
# Run with: uvicorn main:app --reload   
# ---------------------------
# ---------------------------
# Make sure to set up .env with your keys
# ---------------------------