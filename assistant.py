import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
import logging

logging.basicConfig(level=logging.INFO)

# Hugging Face config
HF_API_TOKEN = st.secrets.get("hf_token") or "SEU_TOKEN_HF_AQUI"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Embeddings
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text, max_chunk_size=500):
    chunks = []
    current_chunk = ""
    for sentence in text.split('. '):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embeddings(chunks):
    return EMBEDDING_MODEL.encode(chunks).astype("float32")

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.3,
            "max_new_tokens": 300,
            "do_sample": False,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        st.error(f"Erro da API Hugging Face: {response.text}")
        return "Erro ao gerar resposta."
    return response.json()[0]["generated_text"].strip()

def ask_question(query, chunks, embeddings, faiss_index, top_k=3):
    query_emb = EMBEDDING_MODEL.encode([query]).astype("float32")
    D, I = faiss_index.search(query_emb, top_k)
    context = "\n".join([chunks[i] for i in I[0] if i < len(chunks)])
    prompt = f"Com base no texto abaixo, responda de forma clara e objetiva.\n\nTexto:\n{context}\n\nPergunta: {query}"
    return query_llm(prompt)

def summarize_document(chunks):
    full_text = "\n".join(chunks[:10])  # Limite para não explodir o contexto
    prompt = f"Resuma o conteúdo a seguir de forma clara, objetiva e breve:\n\n{full_text}"
    return query_llm(prompt)

# Streamlit App
st.set_page_config(page_title="Assistente PDF com Mistral", layout="centered")
st.title("📄 Assistente de Leitura de PDF (com Mistral-7B)")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

with st.sidebar:
    st.subheader("📁 Upload de PDF")
    pdf_files = st.file_uploader("Envie arquivos PDF", type=["pdf"], accept_multiple_files=True)
    if st.button("📚 Processar PDFs"):
        if pdf_files:
            with st.spinner("Lendo PDFs..."):
                text = extract_text_from_pdf(pdf_files)
                chunks = get_text_chunks(text)
                embeddings = generate_embeddings(chunks)
                index = create_faiss_index(embeddings)

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.faiss_index = index

            st.success("PDFs processados com sucesso!")
        else:
            st.warning("Envie ao menos um PDF.")

if st.session_state.faiss_index:
    st.markdown("---")
    query = st.chat_input("Digite sua pergunta sobre os PDFs")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Buscando resposta..."):
                answer = ask_question(query, st.session_state.chunks, st.session_state.embeddings, st.session_state.faiss_index)
            st.markdown(f"**Resposta:** {answer}")

    if st.button("📝 Resumir o documento"):
        with st.spinner("Gerando resumo..."):
            summary = summarize_document(st.session_state.chunks)
        st.markdown("### Resumo do Documento:")
        st.info(summary)

st.markdown("---")
st.caption("Made by André Arantes")
