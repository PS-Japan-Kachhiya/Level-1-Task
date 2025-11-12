

import os
import subprocess
import pickle
from urllib.parse import urlparse, parse_qs
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv


load_dotenv()
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.makedirs("processed_videos", exist_ok=True)



def get_video_id(url: str):
    """Extract YouTube video ID from URL."""
    query = parse_qs(urlparse(url).query)
    return query.get("v", ["unknown"])[0]


def save_video_data(video_id, transcription, chunks, chunk_embeddings):
    """Save transcription, chunks, and embeddings for reuse."""
    file_path = f"processed_videos/{video_id}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump({
            "transcription": transcription,
            "chunks": chunks,
            "embeddings": chunk_embeddings
        }, f)
    print(f"💾 Cached data for video {video_id} saved.")


def load_video_data(video_id):
    """Load cached data if already processed."""
    file_path = f"processed_videos/{video_id}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            print(f"📂 Loaded cached data for video {video_id}")
            return pickle.load(f)
    return None


def download_youtube_audio(url: str):
    """Download YouTube audio using yt-dlp."""
    print(f"🎬 Fetching and downloading audio from: {url}")
    try:
        subprocess.run([
            "yt-dlp",
            "-x",
            "--audio-format", "mp3",
            "-o", "%(title)s.%(ext)s",
            "--quiet",
            "--no-warnings",
            url
        ], check=True)

        files = [f for f in os.listdir(".") if f.endswith(".mp3")]
        if not files:
            raise FileNotFoundError("yt-dlp did not produce any mp3 file.")

        latest_file = max(files, key=os.path.getctime)
        print(f"✅ Audio downloaded: {latest_file}")
        return os.path.abspath(latest_file)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ yt-dlp failed: {e}")



def transcribe_audio(youtube_url: str):
    """Transcribe YouTube audio into text using Whisper."""
    print("🎧 Downloading audio for transcription...")
    audio_path = download_youtube_audio(youtube_url)

    print("🧠 Loading Whisper model (base)...")
    whisper_model = whisper.load_model("base")

    print(f"📝 Transcribing: {audio_path}")
    try:
        audio_data = whisper.load_audio(audio_path)
        result = whisper_model.transcribe(audio_data, fp16=False)
        transcription = result["text"].strip()

        print("✅ Transcription complete.")
        return transcription
    except Exception as e:
        print(f"❌ Whisper transcription failed: {e}")
        raise

def split_text(text, chunk_size=1000, overlap=200):
    """Split large text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks



def build_rag_pipeline(youtube_url):
    video_id = get_video_id(youtube_url)
    cached_data = load_video_data(video_id)

    if cached_data:
        transcription = cached_data["transcription"]
        chunks = cached_data["chunks"]
        chunk_embeddings = cached_data["embeddings"]
    else:
        transcription = transcribe_audio(youtube_url)
        chunks = split_text(transcription)

        print("🔍 Generating embeddings...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        chunk_embeddings = embedder.encode(chunks)
        print("✅ Embeddings generated.")

        save_video_data(video_id, transcription, chunks, chunk_embeddings)

    return transcription, chunks, chunk_embeddings



def create_qa_pipeline():
    print("🤖 Loading QA model (Flan-T5)...")
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True
    )


def rag_query(question, chunks, chunk_embeddings, qa_pipeline):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode([question])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    context = "\n\n".join(top_chunks)

    prompt = f"""
You are a knowledgeable AI assistant.
Give a clear, concise answer using only the context below.
If the context doesn’t contain enough info, say so.

Context:
{context}

Question: {question}
Answer:"""

    answer = qa_pipeline(prompt)[0]["generated_text"]
    return answer.strip()

if __name__ == "__main__":
    YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=Q-_D_2NWECE"

    print("🚀 Starting RAG pipeline...")
    transcription, chunks, chunk_embeddings = build_rag_pipeline(YOUTUBE_VIDEO)

    print(f"🗒️ Transcript length: {len(transcription)} chars, Chunks: {len(chunks)}")

    qa_pipeline = create_qa_pipeline()

    while True:
        question = input("\n❓ Ask a question (or 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break
        print("💡 Answer:", rag_query(question, chunks, chunk_embeddings, qa_pipeline))
