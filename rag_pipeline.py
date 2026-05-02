import os
import re
import shutil
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv() 

# ── Streamlit Cloud: load st.secrets into env vars ────────────────
try:
    import streamlit as st
    for key in ("GROQ_API_KEY", "LANGCHAIN_API_KEY",
                "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"):
        if key not in os.environ or not os.environ[key]:
            val = st.secrets.get(key, None)
            if val:
                os.environ[key] = str(val)
except Exception:
    pass

# ── LangSmith Tracing ─────────────────────────────────────────────
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "youtube-qa")
    print("[INFO] LangSmith tracing enabled")

# ── Global caches ─────────────────────────────────────────────────
_whisper_model = None
_embeddings_model = None


def _get_embeddings():
    """Load HuggingFace embedding model once, reuse across videos."""
    global _embeddings_model
    if _embeddings_model is None:
        print("[INFO] Loading local embedding model (first time may download ~80MB)...")
        _embeddings_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[INFO] Local embedding model ready and cached.")
    return _embeddings_model


# ── Step 1: Extract Video ID ──────────────────────────────────────
def get_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL. Please check and try again.")


# ── Step 1b: Fetch Video Metadata ────────────────────────────────
def _fetch_metadata(youtube_url):
    """Use yt-dlp to get video title, channel, description, language, duration, and category."""
    import yt_dlp
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'no_check_certificates': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

    title = info.get('title', 'Unknown Title')
    channel = info.get('uploader', info.get('channel', 'Unknown Channel'))
    description = info.get('description', '')
    if len(description) > 500:
        description = description[:500] + "..."

    duration_sec = info.get('duration', 0)
    if duration_sec:
        hours, remainder = divmod(int(duration_sec), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds}s"
        else:
            duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = 'Unknown'

    language = info.get('language', None)
    if not language:
        subs = info.get('subtitles', {})
        auto_subs = info.get('automatic_captions', {})
        if subs:
            language = list(subs.keys())[0]
        elif auto_subs:
            language = list(auto_subs.keys())[0]
    language = language or 'Unknown'

    lang_map = {
        'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
        'de': 'German', 'pt': 'Portuguese', 'ja': 'Japanese', 'ko': 'Korean',
        'zh': 'Chinese', 'ar': 'Arabic', 'ru': 'Russian', 'it': 'Italian',
        'bn': 'Bengali', 'mr': 'Marathi', 'ta': 'Tamil', 'te': 'Telugu',
        'pa': 'Punjabi', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam',
        'ur': 'Urdu', 'th': 'Thai', 'vi': 'Vietnamese', 'tr': 'Turkish',
    }
    language_full = lang_map.get(language, language)

    categories = info.get('categories', [])
    category = categories[0] if categories else 'Unknown'

    return {
        'title': title,
        'channel': channel,
        'description': description,
        'duration': duration_str,
        'language': language_full,
        'category': category,
    }


# ── Helper: detect Hindi/Devanagari transcript ───────────────────
def _is_hindi_transcript(text):
    """Returns True if >30% of non-space characters are Devanagari."""
    if not text:
        return False
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
    total_chars = len(text.replace(' ', ''))
    if total_chars == 0:
        return False
    return (devanagari_chars / total_chars) > 0.30


# ── Step 2a: Fetch Transcript via YouTube API ─────────────────────
def _fetch_transcript_api(youtube_url):
    """
    Fetch transcript using youtube-transcript-api.
    Priority: manual EN → auto EN → manual HI (translated) → auto HI (translated) → any translatable → first available
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = get_video_id(youtube_url)

    cookies_path = os.path.join(os.path.dirname(__file__), "cookies.txt")
    if os.path.exists(cookies_path):
        from youtube_transcript_api import CookieFileConfig
        api = YouTubeTranscriptApi(cookie_config=CookieFileConfig(cookies_path))
    else:
        api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)
    all_transcripts = list(transcript_list)

    target_transcript = None

    for t in all_transcripts:
        if t.language_code.startswith("en") and not t.is_generated:
            target_transcript = t
            print(f"[INFO] Using manual English subtitles ({t.language_code})")
            break

    if target_transcript is None:
        for t in all_transcripts:
            if t.language_code.startswith("en") and t.is_generated:
                target_transcript = t
                print(f"[INFO] Using auto-generated English captions ({t.language_code})")
                break

    if target_transcript is None:
        for t in all_transcripts:
            if t.language_code.startswith("hi") and not t.is_generated and t.is_translatable:
                target_transcript = t.translate("en")
                print("[INFO] Translating manual Hindi subtitles to English")
                break

    if target_transcript is None:
        for t in all_transcripts:
            if t.language_code.startswith("hi") and t.is_translatable:
                target_transcript = t.translate("en")
                print("[INFO] Translating auto Hindi captions to English")
                break

    if target_transcript is None:
        for t in all_transcripts:
            if t.is_translatable:
                target_transcript = t.translate("en")
                print(f"[INFO] Translating {t.language_code} transcript to English")
                break

    if target_transcript is None:
        if all_transcripts:
            target_transcript = all_transcripts[0]
            print(f"[INFO] Using first available transcript as-is ({all_transcripts[0].language_code})")
        else:
            raise ValueError("No transcripts available via API.")

    fetched = target_transcript.fetch()

    parts = []
    for snippet in fetched.snippets:
        start_sec = int(getattr(snippet, 'start', 0) or 0)
        minutes, seconds = divmod(start_sec, 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        text = snippet.text.strip().replace('\n', ' ')
        if text:
            parts.append(f"{timestamp} {text}")

    full_text = " ".join(parts)
    if not full_text.strip():
        raise ValueError("Transcript fetched but contained no text.")
    return full_text


# ── Helper: Setup ffmpeg ──────────────────────────────────────────
def _setup_ffmpeg():
    """Setup ffmpeg — try static_ffmpeg first, then system ffmpeg."""
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        print("[INFO] static-ffmpeg ready")
        return True
    except Exception as e:
        print(f"[INFO] static-ffmpeg setup failed ({e})")

    import shutil as _shutil
    if _shutil.which("ffmpeg"):
        print("[INFO] System ffmpeg found")
        return True

    print("[WARNING] No ffmpeg found — audio conversion may fail")
    return False


# ── Helper: pytubefix audio download ─────────────────────────────
def _download_audio_pytubefix(youtube_url, temp_dir):
    """
    Download audio using pytubefix — no JS runtime needed, works on Streamlit Cloud.
    Tries multiple strategies to handle bot-detection and age restrictions.
    """
    from pytubefix import YouTube

    strategies = [
        # Strategy 1: Default — simplest, works for most public videos
        {"use_oauth": False, "allow_oauth_cache": False},
        # Strategy 2: With po_token support (newer YouTube bot-detection bypass)
        {"use_oauth": False, "allow_oauth_cache": False, "use_po_token": True},
        # Strategy 3: OAuth cache enabled
        {"use_oauth": False, "allow_oauth_cache": True},
    ]

    last_err = None
    for i, kwargs in enumerate(strategies):
        try:
            print(f"[INFO] pytubefix strategy {i+1}/{len(strategies)}: {kwargs}")
            yt = YouTube(youtube_url, **kwargs)

            # Get lowest-bitrate audio stream to minimise file size / API limits
            stream = (
                yt.streams.filter(only_audio=True, file_extension="mp4").order_by("abr").first()
                or yt.streams.filter(only_audio=True).order_by("abr").first()
                or yt.streams.filter(only_audio=True).first()
            )
            if stream is None:
                raise RuntimeError("No audio streams found")

            print(f"[INFO] Downloading: {stream.mime_type} @ {stream.abr}")
            out_path = stream.download(output_path=temp_dir, filename="audio_raw")

            # Ensure correct extension
            ext = stream.subtype if stream.subtype else "mp4"
            final_path = os.path.join(temp_dir, f"audio.{ext}")
            if os.path.exists(out_path) and out_path != final_path:
                os.rename(out_path, final_path)
            elif not os.path.exists(final_path):
                # Scan temp_dir for the downloaded file
                for f in os.listdir(temp_dir):
                    if not f.endswith(".part"):
                        final_path = os.path.join(temp_dir, f)
                        break

            if os.path.exists(final_path) and os.path.getsize(final_path) > 1000:
                print(f"[INFO] pytubefix strategy {i+1} succeeded: {final_path}")
                return final_path
            else:
                raise RuntimeError(f"Downloaded file missing or too small: {final_path}")

        except Exception as e:
            print(f"[INFO] pytubefix strategy {i+1} failed: {type(e).__name__}: {e}")
            last_err = e
            # Clean temp dir before next attempt
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except Exception:
                    pass
            continue

    raise RuntimeError(f"All pytubefix strategies failed. Last error: {last_err}")


# ── Helper: yt-dlp audio download ────────────────────────────────
def _download_audio_ytdlp(youtube_url, temp_dir):
    """
    Download audio using yt-dlp with multiple client configs.
    Fallback when pytubefix fails. Requires JS runtime for full format support.
    """
    import yt_dlp

    clients = ["ios", "android", "web_creator", "mweb"]
    last_err = None

    for i, client in enumerate(clients):
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'no_check_certificates': True,
            'socket_timeout': 30,
            'extractor_args': {'youtube': {'player_client': [client]}},
        }

        # Use cookies if available
        cookies_path = os.path.join(os.path.dirname(__file__), "cookies.txt")
        if os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
            print(f"[INFO] Using cookies.txt for yt-dlp")

        try:
            print(f"[INFO] yt-dlp attempt {i+1}/{len(clients)} with client={client}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                ext = info.get('ext', 'm4a')
                audio_path = os.path.join(temp_dir, f'audio.{ext}')
                if os.path.exists(audio_path):
                    return audio_path
                # Scan if filename differs
                for f in os.listdir(temp_dir):
                    if f.startswith('audio.') and not f.endswith('.part'):
                        return os.path.join(temp_dir, f)
        except Exception as e:
            print(f"[INFO] yt-dlp client={client} failed: {e}")
            last_err = e
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except Exception:
                    pass

    raise last_err or RuntimeError("yt-dlp: all client configs failed")


# ── Master Audio Downloader ───────────────────────────────────────
def _download_audio_with_retries(youtube_url, temp_dir):
    """
    Download audio using pytubefix first (no JS runtime, cloud-friendly),
    then fall back to yt-dlp with multiple client configs.
    """
    # Primary: pytubefix (no JS runtime needed — works on Streamlit Cloud)
    try:
        return _download_audio_pytubefix(youtube_url, temp_dir)
    except Exception as pytube_err:
        print(f"[INFO] pytubefix failed: {pytube_err}")
        print("[INFO] Falling back to yt-dlp...")

    # Fallback: yt-dlp
    return _download_audio_ytdlp(youtube_url, temp_dir)


# ── Step 2b: Fetch Transcript via Groq Whisper API ───────────────
def _fetch_transcript_groq_whisper(youtube_url, detected_lang=None):
    """
    Download audio and transcribe via Groq Whisper API.
    Uses pytubefix for download (no JS runtime needed on Streamlit Cloud).
    """
    from groq import Groq

    _setup_ffmpeg()

    temp_dir = os.path.join(os.path.dirname(__file__), "_temp_audio")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        print("[INFO] Starting audio download for Groq Whisper...")
        audio_path = _download_audio_with_retries(youtube_url, temp_dir)

        file_size = os.path.getsize(audio_path)
        print(f"[INFO] Audio downloaded: {file_size / (1024*1024):.1f} MB")

        if file_size > 25 * 1024 * 1024:
            raise ValueError(
                f"Audio file too large for Groq API "
                f"({file_size / (1024*1024):.1f} MB > 25 MB limit). "
                "Try a shorter video."
            )

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Language hint for Whisper accuracy
        whisper_lang = None
        if detected_lang:
            lang_lower = detected_lang.lower()
            if lang_lower in ("hi", "hindi"):
                whisper_lang = "hi"
            elif lang_lower in ("en", "english"):
                whisper_lang = "en"

        api_kwargs = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
            "temperature": 0.0,
        }
        if whisper_lang:
            api_kwargs["language"] = whisper_lang

        print(f"[INFO] Sending audio to Groq Whisper API (lang hint: {whisper_lang or 'auto'})...")
        with open(audio_path, "rb") as audio_file:
            api_kwargs["file"] = (os.path.basename(audio_path), audio_file)
            transcription = client.audio.transcriptions.create(**api_kwargs)

        # Build timestamped transcript from segments
        if hasattr(transcription, 'segments') and transcription.segments:
            parts = []
            for seg in transcription.segments:
                text = (
                    seg.get('text', '').strip()
                    if isinstance(seg, dict)
                    else getattr(seg, 'text', '').strip()
                )
                text = text.replace('\n', ' ')
                if not text:
                    continue
                start = (
                    seg.get('start', 0)
                    if isinstance(seg, dict)
                    else getattr(seg, 'start', 0)
                )
                start_sec = int(start)
                minutes, seconds = divmod(start_sec, 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                parts.append(f"{timestamp} {text}")
            if parts:
                return " ".join(parts)

        return transcription.text.strip()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Step 2c: Fetch Transcript via Local Whisper ───────────────────
def _fetch_transcript_audio(youtube_url, detected_lang=None):
    """
    Download audio and transcribe locally with Whisper (last resort).
    task='translate' ensures English output regardless of source language.
    NOTE: openai-whisper + torch are heavy (~2GB). Skipped on Streamlit Cloud.
    """
    global _whisper_model

    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "Local Whisper is not available (openai-whisper not installed). "
            "This is expected on Streamlit Cloud. The video needs YouTube "
            "captions or Groq Whisper to work."
        )

    _setup_ffmpeg()

    temp_dir = os.path.join(os.path.dirname(__file__), "_temp_audio")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        audio_path = _download_audio_with_retries(youtube_url, temp_dir)

        if _whisper_model is None:
            print("[INFO] Loading local Whisper 'base' model...")
            _whisper_model = whisper.load_model("base")

        print("[INFO] Transcribing audio locally (may be slow for long videos)...")
        result = _whisper_model.transcribe(audio_path, task='translate', verbose=False)

        segments = result.get("segments", [])
        if segments:
            parts = []
            for seg in segments:
                text = seg.get('text', '').strip().replace('\n', ' ')
                if not text:
                    continue
                start_sec = int(seg.get('start', 0))
                minutes, seconds = divmod(start_sec, 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                parts.append(f"{timestamp} {text}")
            if parts:
                return " ".join(parts)

        return result["text"].strip()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Step 2: Master Transcript Fetcher ────────────────────────────
def fetch_transcript(youtube_url):
    """
    3-tier transcript fetching:
      Tier 1: YouTube API  — fastest, free, EN + HI with translation
      Tier 2: Groq Whisper — cloud ASR via pytubefix download (no JS runtime)
      Tier 3: Local Whisper — offline fallback (not available on Streamlit Cloud)
    """
    # Detect language hint for Whisper tiers
    detected_lang = None
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({
            'quiet': True, 'skip_download': True, 'no_check_certificates': True
        }) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            detected_lang = info.get('language', None)
            if not detected_lang:
                auto_subs = info.get('automatic_captions', {})
                subs = info.get('subtitles', {})
                keys = list(subs.keys()) + list(auto_subs.keys())
                for k in keys:
                    if k.startswith('hi'):
                        detected_lang = 'hi'
                        break
                    elif k.startswith('en'):
                        detected_lang = 'en'
                if not detected_lang and keys:
                    detected_lang = keys[0]
        print(f"[INFO] Detected video language hint: {detected_lang}")
    except Exception as lang_err:
        print(f"[INFO] Could not detect language ({lang_err}), Whisper will auto-detect.")

    # Tier 1: YouTube transcript API
    try:
        transcript_text = _fetch_transcript_api(youtube_url)
        print("[INFO] Transcript fetched via YouTube API successfully.")
        if _is_hindi_transcript(transcript_text):
            print("[INFO] YouTube API returned untranslated Hindi — falling through to Whisper.")
            raise ValueError("Transcript in Hindi script after API fetch, need Whisper translation.")
        return transcript_text
    except Exception as api_err:
        print(f"[INFO] YouTube API tier failed ({type(api_err).__name__}: {api_err})")

    # Tier 2: Groq Whisper cloud API
    groq_err_msg = "Unknown error"
    try:
        print("[INFO] Trying Groq Whisper cloud API...")
        transcript_text = _fetch_transcript_groq_whisper(youtube_url, detected_lang=detected_lang)
        print("[INFO] Transcript fetched via Groq Whisper API successfully.")
        return transcript_text
    except Exception as groq_err:
        import traceback
        groq_err_msg = f"{type(groq_err).__name__}: {groq_err}"
        print(f"[INFO] Groq Whisper API failed ({groq_err_msg})")
        print(f"[DEBUG] Full Groq traceback:\n{traceback.format_exc()}")

    # Tier 3: Local Whisper
    print("[INFO] Falling back to local Whisper transcription...")
    try:
        transcript_text = _fetch_transcript_audio(youtube_url, detected_lang=detected_lang)
        return transcript_text
    except Exception as local_err:
        raise RuntimeError(
            f"All transcription methods failed.\n\n"
            f"1. YouTube API: No captions available.\n"
            f"2. Groq Whisper API: {groq_err_msg}\n"
            f"3. Local Whisper: {str(local_err)}"
        )


# ── Step 3: Sentence-Aware Chunking with Metadata ────────────────
def _split_into_sentences(text):
    """Split text into sentences using timestamps, punctuation, or whitespace."""
    timestamp_parts = re.split(r'\[\d{2}:\d{2}\]', text)
    timestamp_parts = [p.strip() for p in timestamp_parts if p.strip() and len(p.strip()) > 5]
    if len(timestamp_parts) > 3:
        return timestamp_parts

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    if sentences:
        return sentences

    parts = re.split(r'\s{2,}|\n+', text)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]
    return parts


def split_text_with_metadata(transcript_text, metadata):
    """
    Create sentence-aware chunks with video metadata prepended to every chunk.
    Ensures the LLM always knows the video context regardless of which chunks are retrieved.
    """
    metadata_header = (
        f"[VIDEO INFO] Title: {metadata.get('title', 'Unknown')} | "
        f"Channel: {metadata.get('channel', 'Unknown')} | "
        f"Description: {metadata.get('description', '')[:200]}\n\n"
    )

    sentences = _split_into_sentences(transcript_text)

    if not sentences:
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        fallback_docs = fallback_splitter.create_documents([transcript_text])
        chunks = []
        for doc in fallback_docs:
            chunk_text = metadata_header + doc.page_content
            chunks.append(Document(page_content=chunk_text))
        full_metadata_chunk = (
            f"[VIDEO INFO] This video is titled '{metadata.get('title', 'Unknown')}' "
            f"and is published on the channel '{metadata.get('channel', 'Unknown')}'. "
            f"Video description: {metadata.get('description', 'No description available.')}"
        )
        chunks.insert(0, Document(page_content=full_metadata_chunk))
        return chunks

    target_chunk_size = 1000
    overlap_sentences = 2

    chunks = []
    start = 0
    while start < len(sentences):
        current_chunk_sentences = []
        current_length = 0
        end = start

        while end < len(sentences) and current_length + len(sentences[end]) < target_chunk_size:
            current_chunk_sentences.append(sentences[end])
            current_length += len(sentences[end]) + 1
            end += 1

        if not current_chunk_sentences and end < len(sentences):
            current_chunk_sentences.append(sentences[end])
            end += 1

        chunk_text = metadata_header + " ".join(current_chunk_sentences)
        chunks.append(Document(page_content=chunk_text))

        advance = max(1, len(current_chunk_sentences) - overlap_sentences)
        start += advance

    full_metadata_chunk = (
        f"[VIDEO INFO] This video is titled '{metadata.get('title', 'Unknown')}' "
        f"and is published on the channel '{metadata.get('channel', 'Unknown')}'. "
        f"Video description: {metadata.get('description', 'No description available.')}"
    )
    chunks.insert(0, Document(page_content=full_metadata_chunk))
    return chunks


# ── Step 4: Create Embeddings + FAISS Vector Store ───────────────
def create_vector_store(chunks):
    """
    Embed chunks into FAISS vector store using local HuggingFace embeddings.
    Filters out empty/garbled chunks while accepting both English and Devanagari.
    """
    valid_chunks = []
    for c in chunks:
        if not c.page_content:
            continue
        content = c.page_content
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', content))
        if english_chars >= 5 or devanagari_chars >= 10:
            valid_chunks.append(c)

    if not valid_chunks:
        raise ValueError(
            "No valid text chunks to embed. The transcript may be empty or garbled."
        )

    print(f"[DEBUG] Embedding {len(valid_chunks)} valid chunks (filtered from {len(chunks)})")

    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(valid_chunks, embeddings)
    return vector_store, valid_chunks


# ── Step 5: Build QA Chain with Hybrid Retrieval ─────────────────
def build_qa_chain(vector_store, chunks):
    """
    Build Conversational QA chain with Hybrid Retrieval (BM25 + FAISS).

    Architecture:
    - FAISS (semantic search): finds chunks whose MEANING matches the question
    - BM25 (keyword search): finds chunks that LITERALLY CONTAIN the words
    - EnsembleRetriever: merges both results (0.5 weight each)
    - ConversationalRetrievalChain: rewrites pronoun questions using chat history
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    faiss_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 12, "lambda_mult": 0.7}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    print("[INFO] Hybrid retriever ready (FAISS semantic + BM25 keyword)")

    prompt_template = """You are an intelligent video Q&A assistant. You answer questions about a YouTube video using ONLY the Context provided below.

ANSWERING RULES:
1. ALWAYS TRY TO ANSWER using the Context. If the Context contains ANY relevant information — even indirectly — use it to construct a helpful answer.
2. Only say "This topic is not discussed in the video" when the question is COMPLETELY unrelated to anything in the Context.
3. If the Context mentions a place, person, event, or detail that relates to the question, USE IT to answer — even if the answer is inferred.

RESPONSE LENGTH POLICY (ADAPTIVE):
- Choose answer length based on the question:
  - SHORT (1-3 bullet points): for simple factual questions (who/where/when/yes-no/direct fact).
  - MEDIUM (4-7 bullet points or short numbered list): for "what", "how", "why" questions needing explanation.
  - DETAILED (structured sections): for broad, comparative, summary, or multi-part questions.
- If the user explicitly asks "brief", "short", "in one line", keep it SHORT.
- If the user asks "detailed", "deep", "explain fully", use DETAILED.

FORMATTING POLICY (MARKDOWN):
- Use clean Markdown formatting when useful:
  - Bullets (`-`) for key points.
  - Numbered list (`1.`, `2.`, `3.`) for steps, sequence, ranking, or comparisons.
  - `**bold**` for important facts/names/conclusions.
  - `*italic*` for nuance, caveats, or soft inferences.
- Do NOT over-format. Keep structure readable and natural.
- For very short factual answers, 1-2 plain sentences are allowed.

CONTENT QUALITY RULES:
- When asked about a PERSON: describe who they are based on what they say/do in the video, not just a name.
- When asked follow-ups with pronouns (he/she/they), resolve from context and answer specifically.
- When asked about a TOPIC: explain what was said, who said it, and why it matters in that video.
- If inferring (example: origin/place), clearly phrase as: "Based on the video, X appears to...".
- If the question asks "what is this video about?", provide a clear, structured summary.
- Avoid repetition. Each answer should add value.

CITATIONS (MANDATORY):
- Include [MM:SS] timestamps for factual claims, quotes, or events taken from transcript context.
- Attach the timestamp right after the supported statement.
- If multiple points come from different moments, cite each point separately.

STRICT BOUNDARIES:
- NEVER answer questions about yourself (name/model/training). Reply: "I can only answer questions about this video."
- For truly unrelated topics (math, coding, recipes, general knowledge not in this video): Reply: "I can only answer questions about this video. That topic is not discussed here."
- NEVER fabricate facts not supported by Context.

Context from video:
{context}

User Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        """Given the following Chat History and a Follow Up question, rewrite the Follow Up question as a standalone question that can be understood without the chat history.

RULES:
- If the Follow Up contains pronouns (he, she, it, they, his, her, their), replace them with the actual names or subjects from the Chat History.
- If the Follow Up says "who is he?" or "who is she?", look at the Chat History to find the person being discussed and rewrite it as "who is [that person's name]?"
- If the Follow Up asks about "his girlfriend" or "her boyfriend", find the relevant names from Chat History.
- If there's no Chat History or the question is already standalone, return it unchanged.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False,
        verbose=True
    )
    return qa_chain


# ── Step 6: Auto Chapter Generator ───────────────────────────────
def generate_chapters(transcript_text, metadata):
    """
    Generate automatic video chapters from the transcript using the LLM.
    Samples chunks evenly across the video for full coverage with minimal tokens.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    segments = re.split(r'(\[\d{2}:\d{2}\])', transcript_text)

    timestamped_parts = []
    i = 0
    while i < len(segments):
        if re.match(r'\[\d{2}:\d{2}\]', segments[i]):
            ts = segments[i]
            text = segments[i + 1].strip() if i + 1 < len(segments) else ""
            if text:
                timestamped_parts.append(f"{ts} {text[:150]}")
            i += 2
        else:
            i += 1

    total = len(timestamped_parts)
    if total <= 20:
        sampled = timestamped_parts
    else:
        step = total // 20
        sampled = [timestamped_parts[i] for i in range(0, total, step)][:20]

    if not sampled:
        return "Could not generate chapters — transcript has no timestamps."

    sampled_text = "\n".join(sampled)

    prompt = f"""You are analyzing a YouTube video transcript to create chapter markers.

Video Title: {metadata.get('title', 'Unknown')}
Channel: {metadata.get('channel', 'Unknown')}

Below are timestamped excerpts from across the video. Based on these, create 6-10 chapter markers that divide the video into logical sections.

FORMAT — use EXACTLY this format for each chapter:
[MM:SS] Chapter Title — One sentence description

RULES:
- Use timestamps that appear in the transcript, do NOT invent timestamps
- Chapter titles should be short (3-7 words)
- Descriptions should be one clear sentence
- Cover the video from start to end
- Group related topics into single chapters

Transcript excerpts:
{sampled_text}

Chapters:"""

    response = llm.invoke(prompt)
    return response.content.strip()


# ── Master Function: URL → Ready QA Chain ────────────────────────
def process_youtube_url(url):
    # 1. Fetch metadata
    t0 = time.time()
    try:
        metadata = _fetch_metadata(url)
        print(f"[INFO] Video: {metadata['title']}")
    except Exception as meta_err:
        print(f"[INFO] Metadata fetch failed ({meta_err}), using defaults")
        metadata = {
            'title': 'Unknown', 'channel': 'Unknown', 'description': '',
            'duration': 'Unknown', 'language': 'Unknown', 'category': 'Unknown',
        }
    print(f"[TIMING] Metadata: {time.time() - t0:.1f}s")

    # 2. Fetch transcript
    t1 = time.time()
    transcript = fetch_transcript(url)
    print(f"[DEBUG] Transcript length: {len(transcript)} chars")
    print(f"[TIMING] Transcript: {time.time() - t1:.1f}s")

    if not transcript or len(transcript.strip()) < 50:
        raise ValueError(
            "Could not extract a usable transcript from this video. "
            "The video may not have captions, or the audio transcription produced no text."
        )

    words = re.findall(r'[a-zA-Z\u0900-\u097F]{2,}', transcript)
    print(f"[DEBUG] Real word count: {len(words)}")
    if len(words) < 20:
        raise ValueError(
            "The transcript has very little readable content. "
            "This may happen if the video language is not well supported by the transcription model."
        )

    # 3. Sentence-aware chunking with metadata on every chunk
    t2 = time.time()
    chunks = split_text_with_metadata(transcript, metadata)
    print(f"[INFO] Created {len(chunks)} chunks (sentence-aware, metadata-enriched)")
    print(f"[TIMING] Chunking: {time.time() - t2:.1f}s")

    if len(chunks) <= 1:
        raise ValueError(
            "Transcript was too short to create meaningful chunks. "
            "The video may have very little spoken content."
        )

    # 4. Build vector store
    t3 = time.time()
    try:
        vector_store, valid_chunks = create_vector_store(chunks)
    except Exception as embed_err:
        print(f"[ERROR] Embedding failed: {embed_err}")
        raise ValueError(
            "Failed to process the transcript text. "
            "This may happen with very long videos or unsupported languages. "
            "Try a shorter video or one with English subtitles."
        )
    print(f"[TIMING] Embedding + FAISS: {time.time() - t3:.1f}s")

    # 5. Build QA chain with Hybrid Retrieval
    t4 = time.time()
    qa_chain = build_qa_chain(vector_store, valid_chunks)
    print(f"[TIMING] QA Chain + BM25: {time.time() - t4:.1f}s")

    # 6. Extract video_id for timestamp links
    video_id = get_video_id(url)

    print(f"[TIMING] TOTAL: {time.time() - t0:.1f}s")
    return qa_chain, len(chunks), metadata, transcript, video_id