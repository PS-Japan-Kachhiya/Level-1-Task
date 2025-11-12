# import streamlit as st
# import os
# from rag import transcribe_audio, split_text, retrieve_context, rag_query, SentenceTransformer, cosine_similarity

# st.set_page_config(page_title="🎥 YouTube RAG Assistant", layout="wide")
# st.title("🎧 YouTube RAG Assistant")
# st.caption("Upload a YouTube URL → Transcribe → Ask Questions about the video.")


# youtube_url = st.text_input("🔗 Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
# question = st.text_input("❓ Ask a question about the video:", placeholder="What is this video about?")


# if st.button("Process Video"):
#     if not youtube_url:
#         st.warning("Please enter a YouTube video URL.")
#     else:
#         with st.spinner("Downloading and transcribing audio..."):
#             try:
#                 transcription = transcribe_audio(youtube_url)
#                 st.success("✅ Transcription complete!")
#                 st.text_area("📝 Transcribed Text", transcription[:3000] + "..." if len(transcription) > 3000 else transcription, height=300)

#                 with st.spinner("Generating embeddings..."):
#                     chunks = split_text(transcription)
#                     embedder = SentenceTransformer("all-MiniLM-L6-v2")
#                     chunk_embeddings = embedder.encode(chunks)
#                     st.success(f"✅ Split into {len(chunks)} chunks and embeddings generated.")

#                 st.session_state["chunks"] = chunks
#                 st.session_state["embedder"] = embedder
#                 st.session_state["chunk_embeddings"] = chunk_embeddings

#             except Exception as e:
#                 st.error(f"❌ Error during transcription: {e}")

# if "chunks" in st.session_state and question:
#     with st.spinner("Retrieving context and generating answer..."):
#         try:
#             query_embedding = st.session_state["embedder"].encode([question])
#             similarities = cosine_similarity(query_embedding, st.session_state["chunk_embeddings"])[0]
#             top_indices = similarities.argsort()[-3:][::-1]
#             top_chunks = [st.session_state["chunks"][i] for i in top_indices]
#             context = "\n\n".join(top_chunks)

#             answer = rag_query(question)
#             st.subheader("💡 Answer")
#             st.write(answer)

           
#         except Exception as e:
#             st.error(f"❌ Failed to generate answer: {e}")
# app.py
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rag import transcribe_audio, split_text, rag_query, create_qa_pipeline

st.set_page_config(page_title="🎥 YouTube RAG Assistant", layout="wide")
st.title("🎧 YouTube RAG Assistant")
st.caption("Enter YouTube URL → Process once → Ask multiple questions")

# Inputs
youtube_url = st.text_input("🔗 Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
question = st.text_input("❓ Ask a question about the video:", placeholder="What is this video about?")

# Make sure session_state keys exist
if "chunks" not in st.session_state:
    st.session_state["chunks"] = None
if "chunk_embeddings" not in st.session_state:
    st.session_state["chunk_embeddings"] = None
if "embedder" not in st.session_state:
    st.session_state["embedder"] = None
if "video_url" not in st.session_state:
    st.session_state["video_url"] = None
if "qa_pipeline" not in st.session_state:
    # create or lazy-load the QA model pipeline once
    try:
        st.session_state["qa_pipeline"] = create_qa_pipeline()
    except Exception as e:
        st.warning("QA model not preloaded: it will be loaded on demand.")
        st.session_state["qa_pipeline"] = None

# Process Video button
if st.button("🚀 Process Video"):
    if not youtube_url:
        st.warning("Please enter a YouTube video URL.")
    else:
        with st.spinner("Downloading/transcribing and creating embeddings (one-time)..."):
            try:
                # 1) Transcribe
                transcription = transcribe_audio(youtube_url)
                st.success("✅ Transcription complete!")
                st.text_area("📝 Transcribed Text", transcription[:3000] + "..." if len(transcription) > 3000 else transcription, height=300)

                # 2) Split & embed
                chunks = split_text(transcription)
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                chunk_embeddings = embedder.encode(chunks)

                # 3) Save in session
                st.session_state["chunks"] = chunks
                st.session_state["chunk_embeddings"] = chunk_embeddings
                st.session_state["embedder"] = embedder
                st.session_state["video_url"] = youtube_url

                st.success(f"✅ Split into {len(chunks)} chunks and embeddings generated.")
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

# Answer question
if st.session_state["chunks"] and question:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            # Ensure question is for the same processed video
            if st.session_state.get("video_url") != youtube_url:
                st.warning("⚠️ This question doesn't match the last processed video. Click 'Process Video' for this URL first.")
            else:
                # Local retrieval (top-k)
                embedder = st.session_state["embedder"]
                query_emb = embedder.encode([question])
                sims = cosine_similarity(query_emb, st.session_state["chunk_embeddings"])[0]
                top_indices = sims.argsort()[-3:][::-1]
                top_chunks = [st.session_state["chunks"][i] for i in top_indices]
                context = "\n\n".join(top_chunks)

                # Ensure QA pipeline is loaded (lazy)
                if st.session_state["qa_pipeline"] is None:
                    st.session_state["qa_pipeline"] = create_qa_pipeline()

                # Call rag_query with required args
                answer = rag_query(question, st.session_state["chunks"], st.session_state["chunk_embeddings"], st.session_state["qa_pipeline"])

                st.subheader("💡 Answer")
                st.write(answer)

                with st.expander("📚 Context used"):
                    st.write(context)

        except Exception as e:
            st.error(f"❌ Failed to generate answer: {e}")
