from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Optional if running locally

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR/ "templates")


class Message(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask(msg: Message):
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": msg.message, "stream": False}
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        reply = result.get("response", "No reply.")
        return {"reply": reply.strip()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"reply": f"Error: {str(e)}"})
