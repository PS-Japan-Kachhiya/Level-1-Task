 ## 🎧 YouTube RAG Assistant

This project is a **Retrieval-Augmented Generation (RAG)** pipeline built using **Streamlit**, **Whisper**, and **Hugging Face Transformers**.  
It allows you to:
- Input any **YouTube video link** 🎥  
- Automatically **download and transcribe** the audio using **OpenAI Whisper**  
- Use **sentence embeddings** to create contextual knowledge chunks  
- Ask **natural language questions** about the video content and get AI-generated answers 💬  



 ## 🚀 Features

✅ Download YouTube video audio  
✅ Transcribe audio → text using **Whisper**  
✅ Create embeddings with **Sentence Transformers**  
✅ Retrieve the most relevant context for your question  
✅ Generate accurate, summarized answers using **Flan-T5**  
✅ Interactive **Streamlit UI**  



## 🧱 Project Structure
```plaintext
youtube-rag-assistant/
│
├── rag.py                 # Core RAG pipeline (transcription, embedding, QA)
├── app.py                 # Streamlit UI for interaction
├── requirements.txt       # Dependencies
└── README.md              # Project overview and usage guide
```



## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
git clone https://github.com/PS-Japan-Kachhiya/Level-1-Task.git

### 2️⃣ Create a Virtual Environment
python -m venv .venv
Activate it

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run the Application
streamlit run app.py

