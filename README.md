# 🎙️ ARIA – Voice AI Assistant

**Speech-to-Speech AI assistant** built with Faster-Whisper + Groq LLaMA + gTTS  
Pipeline: `Your voice → Faster-Whisper STT → LLM (Groq) → gTTS TTS → Voice response`

🌐 **Live Demo:** [ayush-s-tomar.github.io/aria-voice-assistant](https://ayush-s-tomar.github.io/aria-voice-assistant)  
⚙️ **Backend API:** [aria-voice-assistant-6eze.onrender.com](https://aria-voice-assistant-6eze.onrender.com)  
📖 **API Docs:** [aria-voice-assistant-6eze.onrender.com/docs](https://aria-voice-assistant-6eze.onrender.com/docs)

---

## ✨ Features

- 🎤 **Voice input** — record directly from your browser mic
- 🌍 **99-language support** — speak in Hindi, Spanish, French, English, and more — auto-detected
- 🧠 **Conversation memory** — ARIA remembers context across turns in a session
- 🔊 **Voice output** — responses spoken aloud via gTTS (or ElevenLabs for premium voice)
- ⚡ **Fast inference** — Groq LLaMA-3.3-70B for near-instant responses
- 💬 **Text fallback** — type messages if mic isn't available

---

## 🗂️ Project Structure

```
aria-voice-assistant/
├── backend/
│   ├── main.py                  # FastAPI app (3 endpoints)
│   ├── requirements.txt
│   ├── .env.example
│   └── services/
│       ├── transcriber.py       # Faster-Whisper STT (99 languages)
│       ├── llm.py               # Groq LLaMA with rolling memory
│       └── tts.py               # gTTS (free) / ElevenLabs (premium)
├── frontend/
│   └── index.html               # Single-file voice UI (no framework)
├── docs/                        # GitHub Pages deployment
├── render.yaml                  # One-click Render deploy config
└── README.md
```

---

## 🔋 Tech Stack

| Layer | Tech |
|-------|------|
| STT | Faster-Whisper (local, free, 99 languages) |
| LLM | Groq + LLaMA-3.3-70B |
| TTS | gTTS (free) / ElevenLabs (premium) |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deploy | Render (backend) + GitHub Pages (frontend) |

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11
- Groq API key → [console.groq.com](https://console.groq.com)
- ffmpeg installed

```powershell
# Install ffmpeg (Windows)
winget install ffmpeg
```

### Step 1 — Clone & Setup

```powershell
git clone https://github.com/ayush-s-tomar/aria-voice-assistant.git
cd aria-voice-assistant/backend

py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Configure Environment

```powershell
copy .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
WHISPER_MODEL=base        # tiny | base | small | medium
ELEVENLABS_API_KEY=       # optional — leave blank to use free gTTS
```

### Step 3 — Run

```powershell
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in Chrome (double-click or drag into browser).

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/voice` | Full pipeline: audio → STT → LLM → TTS → audio |
| POST | `/chat/text` | Text-only: message → LLM response |
| DELETE | `/session/{id}` | Clear conversation memory |

---

## 🚀 Deploy Your Own

### Backend → Render
1. Fork this repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your fork — `render.yaml` is auto-detected
4. Add env var: `GROQ_API_KEY` in Render dashboard
5. Set `WHISPER_MODEL=tiny` (recommended for free tier)
6. Deploy

### Frontend → GitHub Pages
1. Update `const API` in `frontend/index.html` with your Render URL
2. Copy to `docs/index.html` and push
3. Enable GitHub Pages → branch: `main` → folder: `/docs`

---

## 🧠 How Memory Works

Each browser tab generates a unique `session_id`. The backend maintains a rolling 20-message history per session in memory. Cleared on page refresh or via the "Clear chat" button.

---

## 🔧 Upgrade to Premium Voice (ElevenLabs)

1. Get API key at [elevenlabs.io](https://elevenlabs.io)
2. Add to `.env`:
   ```
   ELEVENLABS_API_KEY=your_key_here
   ```
3. Restart backend — switches automatically, no code changes needed

---

## 🛠️ Built By

**Ayush Singh Tomar** — AI Developer  
[LinkedIn](https://linkedin.com/in/ayushsinghtomar) • [GitHub](https://github.com/ayush-s-tomar) • [Portfolio Projects](https://agentloop.onrender.com)
