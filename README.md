# 🎬 YouTube Video Q&A System

An AI-powered Streamlit application that lets you ask questions about any YouTube video. It uses RAG (Retrieval-Augmented Generation) to extract transcripts, build a searchable knowledge base, and generate accurate, timestamped answers.

---

## ✨ Features

- **Any YouTube URL** — Paste a link and the system fetches metadata + transcript automatically
- **3-Tier Transcription** — YouTube captions → Groq Whisper API → Local Whisper fallback
- **Hybrid Retrieval** — Combines FAISS semantic search + BM25 keyword search for maximum accuracy
- **Timestamped Answers** — Responses include clickable `[MM:SS]` timestamps linking back to the video
- **Conversational Memory** — Ask follow-up questions with pronoun resolution (e.g., "What did he say about...?")
- **Auto Chapter Generation** — Generate a table of contents with timestamps for any video
- **Adaptive Response Length** — Short answers for simple questions, detailed breakdowns for complex ones
- **Multi-Language Support** — Handles English, Hindi, and other languages with automatic translation
- **Off-Topic Filtering** — Rejects unrelated questions to keep responses video-focused

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM** | Groq (`llama-3.1-8b-instant`) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) — runs locally, no API needed |
| **Vector Store** | FAISS |
| **Keyword Search** | BM25 |
| **Audio Download** | pytubefix (primary) + yt-dlp (fallback) |
| **Transcription** | youtube-transcript-api, Groq Whisper API, Local Whisper |
| **Orchestration** | LangChain |

---

## 📁 Project Structure

```
├── app.py                # Streamlit UI — chat interface, video info cards, chapter generator
├── rag_pipeline.py       # Core pipeline — transcript fetching, chunking, embeddings, QA chain
├── requirements.txt      # Python dependencies
├── packages.txt          # System packages for Streamlit Cloud (ffmpeg)
├── .gitignore            # Files excluded from version control
└── README.md             # This file
```

---

## ⚙️ How It Works

```
YouTube URL
    │
    ▼
┌─────────────────┐
│ Fetch Metadata   │  ← yt-dlp (title, channel, duration, language)
└────────┬────────┘
         ▼
┌─────────────────────────────────────────────────────┐
│ Fetch Transcript (3-tier fallback)                   │
│  1. YouTube Captions API (fastest, free)             │
│  2. Groq Whisper API (cloud ASR via audio download)  │
│  3. Local Whisper (offline, needs openai-whisper)     │
└────────┬────────────────────────────────────────────┘
         ▼
┌─────────────────┐
│ Chunk Transcript │  ← Sentence-aware splitting with video metadata headers
└────────┬────────┘
         ▼
┌─────────────────┐
│ Embed & Index    │  ← HuggingFace all-MiniLM-L6-v2 → FAISS vector store
└────────┬────────┘
         ▼
┌──────────────────────────────┐
│ Hybrid Retrieval             │  ← FAISS (semantic) + BM25 (keyword)
│ EnsembleRetriever (0.5/0.5)  │
└────────┬─────────────────────┘
         ▼
┌─────────────────┐
│ Generate Answer  │  ← Groq LLM with timestamped citations
└─────────────────┘
```

---

## 🚀 Local Setup

### Prerequisites

- Python 3.11+
- Git
- [Groq API Key](https://console.groq.com) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/yashvipatel1311/YouTube-Video-Q-A-System.git
cd YouTube-Video-Q-A-System
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create `.env` file

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional — LangSmith tracing for debugging
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=youtube-qa
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment

### Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app
3. Set the main file to `app.py`
4. Add your secrets in **Settings → Secrets** using TOML format:
   ```toml
   GROQ_API_KEY = "your_key_here"
   LANGCHAIN_API_KEY = "your_key_here"
   LANGCHAIN_TRACING_V2 = "true"
   LANGCHAIN_PROJECT = "youtube-qa"
   ```
5. Deploy

### Hugging Face Spaces

1. Create a new Space with **Streamlit** SDK
2. Link your GitHub repo or upload files directly
3. Add secrets in **Settings → Variables and Secrets**
4. The app will build and deploy automatically

### Render / Railway

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
- Add environment variables in the platform dashboard

---

## 🔑 API Keys

| Key | Required | Where to Get |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) |
| `LANGCHAIN_API_KEY` | ❌ Optional | [smith.langchain.com](https://smith.langchain.com) — for tracing/debugging |

---

## 📝 Notes

- Videos with existing subtitles/captions process fastest
- For videos without captions, audio is downloaded and transcribed via Groq Whisper API
- First run downloads the embedding model (~80 MB) — subsequent runs use the cached model
- Long videos create more chunks and may take longer to process

---

## 🔒 Security

- Never commit your `.env` file
- API keys should be set via environment variables or platform secrets
- Rotate any accidentally exposed keys immediately

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
