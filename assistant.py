import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging
from transformers import AutoTokenizer

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar modelo de embeddings
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Função para extrair texto e metadados de PDFs
def extract_text_from_pdf(pdf_files):
    logger.info("Extraindo texto de %d PDFs", len(pdf_files))
    text = ""
    file_names = []
    authors = []
    titles = []
    for pdf in pdf_files:
        try:
            file_names.append(pdf.name)
            reader = PdfReader(pdf)
            metadata = reader.metadata or {}
            authors.append(metadata.get('/Author', 'Desconhecido'))
            titles.append(metadata.get('/Title', 'Desconhecido'))
            for page in reader.pages:
                extracted_text = page.extract_text() or ""
                text += extracted_text + "\n"
        except Exception as e:
            logger.error("Erro ao extrair texto do PDF %s: %s", pdf.name, e)
            st.error(f"Erro ao extrair texto do PDF {pdf.name}: {e}")
    logger.info("Extração de texto finalizada. Tamanho: %d caracteres", len(text))
    return text, file_names, authors, titles

# Função para dividir texto em chunks
def get_text_chunks(text, max_chunk_size=300):
    logger.info("Dividindo texto em chunks...")
    chunks = []
    current_chunk = ""
    for sentence in text.split('. '):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    chunks = [chunk for chunk in chunks if chunk]
    logger.info("Número de chunks gerados: %d", len(chunks))
    return chunks

# Função para gerar embeddings
def generate_embeddings(text_chunks):
    if not text_chunks:
        logger.warning("Nenhum chunk para gerar embeddings.")
        return np.array([])
    logger.info("Gerando embeddings para %d textos...", len(text_chunks))
    try:
        embeddings = EMBEDDING_MODEL.encode(text_chunks, show_progress_bar=False)
        return embeddings.astype('float32')
    except Exception as e:
        logger.error("Erro ao gerar embeddings: %s", e)
        st.error(f"Erro ao gerar embeddings: {e}")
        return np.array([])

# Função para criar índice FAISS
def create_faiss_index(embeddings):
    if embeddings.size == 0:
        logger.warning("Nenhum embedding para criar o índice FAISS.")
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logger.info("Índice FAISS criado com sucesso.")
    return index

# Função para truncar contexto com base em tokens
def truncate_context(context, question, max_tokens=800):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    prompt_template = """Você é um assistente útil. Responda em português, de forma breve e precisa. Para perguntas sobre nome do arquivo, autor ou título, use o metadata. Para resumo, resuma em até 3 frases. Se não souber, diga "Não sei".

[Contexto]
{context}

[Pergunta]
{question}

[Resposta]
"""
    base_prompt = prompt_template.format(context="", question=question)
    base_tokens = len(tokenizer.encode(base_prompt))
    remaining_tokens = max_tokens - base_tokens
    context_tokens = tokenizer.encode(context)[:remaining_tokens]
    truncated_context = tokenizer.decode(context_tokens, skip_special_tokens=True)
    return truncated_context

# Função para carregar o modelo de linguagem
@st.cache_resource
def load_llm():
    logger.info("Carregando modelo distilgpt2...")
    try:
        llm = HuggingFacePipeline.from_model_id(
            model_id="distilgpt2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 200, "temperature": 0.05, "do_sample": True}
        )
        logger.info("LLM carregado com sucesso.")
        return llm
    except Exception as e:
        logger.error("Erro ao carregar LLM: %s", e)
        st.error(f"Erro ao carregar LLM: {e}")
        raise

# Função para gerar resposta com LLM
def generate_response(query, context, file_names, authors, titles):
    if not context:
        return "Não consegui encontrar informações relevantes nos documentos."
    
    context_str = "\n".join(context)
    context_str = truncate_context(context_str, query, max_tokens=800)
    
    prompt_template = ChatPromptTemplate.from_template(
        f"Com base nos documentos, sobre '{query}', encontrei as seguintes informações:\n\n{context_str}\n\n" \
        f""
    )
    
    llm = load_llm()
    chain = prompt_template | llm
    
    # Verificar perguntas sobre metadados
    if "nome do arquivo" in query.lower() and file_names:
        return f"O nome do arquivo é: {file_names[0]}"
    elif "autor" in query.lower() and authors:
        return f"O autor do arquivo é: {authors[0]}"
    elif "título" in query.lower() and titles:
        response = f"O título do arquivo é: {titles[0]}"
        if "resum" in query.lower():
            summary = chain.invoke({"context": context_str, "input": query}).strip()
            summary = summary.replace("[Resposta]", "").strip() or "Não sei"
            return f"{response}\n\nResumo: {summary}"
        return response
    
    # Gerar resposta com LLM
    response = chain.invoke({"context": context_str, "input": query}).strip()
    response = response.replace("[Resposta]", "").strip() or "Não sei"
    return response

# Configuração do Streamlit
st.set_page_config(page_title="Tarefa AS05", layout="centered")
st.title("📚 Assistente AS05 - Pergunte sobre um PDF!")
st.markdown("Faça upload de seus PDFs e faça perguntas sobre o conteúdo deles.")

# Inicializar estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "authors" not in st.session_state:
    st.session_state.authors = []
if "titles" not in st.session_state:
    st.session_state.titles = []

# Exibir mensagens do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de perguntas
user_query = st.chat_input("Faça uma pergunta sobre o PDF")

if user_query:
    if not st.session_state.documents_processed or st.session_state.faiss_index is None:
        st.error("Faça o upload e processe seus PDFs na barra lateral.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        try:
            logger.info("Processando pergunta: %s", user_query)
            query_embedding = EMBEDDING_MODEL.encode([user_query])[0]
            D, I = st.session_state.faiss_index.search(np.array([query_embedding]).astype('float32'), k=3)
            relevant_context = [st.session_state.text_chunks[i] for i in I[0] if i < len(st.session_state.text_chunks)]
            
            with st.spinner("Gerando resposta..."):
                response = generate_response(
                    user_query,
                    relevant_context,
                    st.session_state.file_names,
                    st.session_state.authors,
                    st.session_state.titles
                )
            
            with st.chat_message("assistant"):
                st.markdown(f"**Resposta:**\n{response}")
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.exception("Erro ao gerar resposta: %s", e)
            st.error(f"Erro ao gerar resposta: {e}")

# Sidebar para upload de PDFs
with st.sidebar:
    st.title("Seus Documentos")
    pdf_docs = st.file_uploader(
        "Carregue seus arquivos PDF aqui e clique em 'Processar'",
        accept_multiple_files=True,
        type="pdf",
    )

    if st.button("Processar PDFs"):
        if pdf_docs:
            try:
                with st.spinner("Processando PDFs... Isso pode levar um momento."):
                    logger.info("Iniciando processamento de PDFs.")
                    raw_text, st.session_state.file_names, st.session_state.authors, st.session_state.titles = extract_text_from_pdf(pdf_docs)
                    st.session_state.text_chunks = get_text_chunks(raw_text)
                    st.session_state.embeddings = generate_embeddings(st.session_state.text_chunks)
                    st.session_state.faiss_index = create_faiss_index(st.session_state.embeddings)
                    st.session_state.documents_processed = True
                    st.success(f"✔️ {len(pdf_docs)} PDFs processados com sucesso! Agora você pode fazer perguntas.")
            except Exception as e:
                logger.exception("Erro ao processar PDFs: %s", e)
                st.error(f"Erro ao processar PDFs: {e}")
        else:
            st.warning("Carregue pelo menos um arquivo PDF para processar.")

    if st.session_state.faiss_index is not None and st.button("Resetar e fazer upload de novos PDFs"):
        st.session_state.documents_processed = False
        st.session_state.messages = []
        st.session_state.text_chunks = []
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        st.session_state.file_names = []
        st.session_state.authors = []
        st.session_state.titles = []
        st.rerun()

st.markdown("---")
st.caption("Tarefa AS05 - André")