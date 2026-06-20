# 🎙️ ARIA – Voice AI Assistant

**Speech-to-Speech AI assistant** built with Whisper + Groq LLaMA + gTTS  
Pipeline: `Your voice → Whisper STT → LLM (Groq) → gTTS TTS → Voice response`

---

## 🗂️ Project Structure

```
voice-ai-assistant/
├── backend/
│   ├── main.py                  # FastAPI app (3 endpoints)
│   ├── requirements.txt
│   ├── .env.example
│   └── services/
│       ├── transcriber.py       # Whisper STT
│       ├── llm.py               # Groq LLaMA with memory
│       └── tts.py               # gTTS / ElevenLabs TTS
├── frontend/
│   └── index.html               # Single-file voice UI
├── render.yaml                  # Deploy to Render
└── README.md
```

---

## ⚙️ Step-by-Step Setup (Local)

### Step 1 — Prerequisites
- Python 3.11 (same as your other projects)
- A Groq API key → https://console.groq.com
- ffmpeg installed (needed by Whisper)

**Install ffmpeg on Windows (PowerShell):**
```powershell
winget install ffmpeg
# OR download from https://ffmpeg.org/download.html and add to PATH
```

---

### Step 2 — Clone & Setup Backend

```powershell
cd voice-ai-assistant/backend

# Create virtual env with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** First run will download the Whisper `base` model (~140 MB). This is automatic.

---

### Step 3 — Configure Environment

```powershell
copy .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
WHISPER_MODEL=base
# Leave ELEVENLABS_API_KEY blank to use free gTTS
```

---

### Step 4 — Run the Backend

```powershell
# Make sure venv is active
uvicorn main:app --reload --port 8000
```

Open → http://localhost:8000  
Docs → http://localhost:8000/docs

---

### Step 5 — Open the Frontend

Just open `frontend/index.html` in your browser (double-click or drag into Chrome).

> No server needed for the frontend — it's a single HTML file.

---

### Step 6 — Test It

1. Click the **purple mic button**
2. Speak your message
3. Click again to stop recording
4. Wait 2–3 seconds for ARIA to respond with voice
5. Or type in the text box and press Enter

---

## 🚀 Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Render detects `render.yaml` automatically
5. Add env var: `GROQ_API_KEY` in Render dashboard
6. Deploy → get your public URL
7. Update `API` in `frontend/index.html` line 1:
   ```js
   const API = "https://your-app.onrender.com";
   ```
8. Host the frontend on GitHub Pages (just push `frontend/` to a gh-pages branch)

---

## 🔧 Upgrade: ElevenLabs Premium Voice

1. Get API key at https://elevenlabs.io
2. Add to `.env`:
   ```
   ELEVENLABS_API_KEY=your_key_here
   ```
3. Restart backend — it auto-detects and switches to ElevenLabs

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/voice` | Full pipeline: audio → text → LLM → audio |
| POST | `/chat/text` | Text-only: message → LLM response |
| DELETE | `/session/{id}` | Clear conversation memory |

---

## 🧠 How Memory Works

Each browser tab generates a unique `session_id`. The backend keeps a rolling 20-message history per session. Memory is cleared on page refresh or via the "Clear chat" button.

---

## 🔋 Tech Stack

| Layer | Tech |
|-------|------|
| STT | OpenAI Whisper (local, free) |
| LLM | Groq + LLaMA-3.3-70B |
| TTS | gTTS (free) / ElevenLabs (premium) |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deploy | Render |

---

## 💡 LinkedIn Post Angle

> "Built ARIA — a voice AI assistant that listens, thinks, and talks back.  
> Full speech-to-speech pipeline: Whisper → LLaMA via Groq → gTTS  
> Under 200 lines of Python. Deployed on Render. 🎙️"
