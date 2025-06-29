import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Classe para embeddings usando sentence_transformers
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        logger.info("Gerando embeddings para %d textos via sentence_transformers...", len(texts))
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.astype('float32')
        except Exception as e:
            logger.error("Erro ao gerar embeddings para documentos: %s", e)
            st.error(f"Erro ao gerar embeddings para documentos: {e}")
            return np.zeros((len(texts), self.dimension)).astype('float32')

    def embed_query(self, text):
        logger.info("Gerando embedding para query: %s", text)
        try:
            embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
            return embedding.astype('float32')
        except Exception as e:
            logger.error("Erro ao gerar embedding para query '%s': %s", text[:50], e)
            st.error(f"Erro ao gerar embedding para query: {e}")
            return np.zeros(self.dimension).astype('float32')

# Função para carregar o modelo de linguagem
@st.cache_resource
def load_llm():
    logger.info("Carregando modelo de linguagem distilgpt2...")
    try:
        llm = HuggingFacePipeline.from_model_id(
            model_id="distilgpt2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.1, "do_sample": True}
        )
        logger.info("LLM carregado com sucesso.")
        return llm
    except Exception as e:
        logger.error("Erro ao carregar LLM: %s", e)
        st.error(f"Erro ao carregar LLM: {e}")
        raise

# Função para extrair texto de PDFs
def extract_text_from_pdf(pdf_files):
    logger.info("Extraindo texto de %d PDFs", len(pdf_files))
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                extracted_text = page.extract_text() or ""
                text += extracted_text + "\n"
        except Exception as e:
            logger.error("Erro ao extrair texto do PDF: %s", e)
            st.error(f"Erro ao extrair texto do PDF: {e}")
    logger.info("Extração de texto finalizada. Tamanho: %d caracteres", len(text))
    return text

# Função para dividir texto em chunks
def get_text_chunks(text):
    logger.info("Dividindo texto em chunks...")
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    logger.info("Número de chunks gerados: %d", len(chunks))
    return chunks

# Função para criar o índice FAISS e documentos
def create_faiss_index(embeddings, text_chunks):
    if embeddings.size == 0:
        logger.warning("Nenhum embedding para criar o índice FAISS.")
        return None, None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    documents = [Document(page_content=chunk, metadata={"source": "uploaded_pdf"}) for chunk in text_chunks]
    logger.info("Índice FAISS e documentos criados com sucesso.")
    return index, documents

# Função para criar a cadeia inevitável de conversação
def get_conversational_chain():
    logger.info("Inicializando cadeia de conversação...")
    prompt_template = """Você é um assistente útil. Responda à pergunta com base no contexto, **em português**, de forma breve e precisa.
Se não souber, diga que não sabe.

[Contexto]
{context}

[Pergunta]
{input}

[Resposta]
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = load_llm()
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain

# Função auxiliar para criar retriever
def create_retriever(documents, faiss_index):
    embedding_function = SentenceTransformerEmbeddings()
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    vector_store = FAISS(
        embedding_function=embedding_function,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vector_store.as_retriever()

# Configuração do Streamlit
st.set_page_config(page_title="Tarefa AS05", layout="centered")
st.title("📚 Tarefa AS05 - Pergunte sobre um PDF!")
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
if "documents" not in st.session_state:
    st.session_state.documents = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Exibir mensagens do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de perguntas
user_question = st.chat_input("Faça uma pergunta sobre o PDF")

if user_question:
    if not st.session_state.documents_processed or st.session_state.faiss_index is None:
        st.error("Faça o upload e processe seus PDFs na barra lateral.")
    else:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        try:
            logger.info("Processando pergunta: %s", user_question)
            embedding_function = SentenceTransformerEmbeddings()
            query_embedding = embedding_function.embed_query(user_question)
            D, I = st.session_state.faiss_index.search(np.array([query_embedding]).astype('float32'), k=3)
            relevant_documents = [st.session_state.documents[i] for i in I[0] if i < len(st.session_state.documents)]
            
            document_chain = get_conversational_chain()
            retriever = create_retriever(st.session_state.documents, st.session_state.faiss_index)
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            with st.spinner("Gerando resposta..."):
                response = rag_chain.invoke({"input": user_question})
                response_text = response["answer"].strip()
                # Limpar a resposta, removendo qualquer prefixo indesejado
                if response_text.startswith("[Resposta]"):
                    response_text = response_text[len("[Resposta]"):].strip()

            with st.chat_message("assistant"):
                st.markdown(f"**Resposta:**\n{response_text}")
            st.session_state.messages.append({"role": "assistant", "content": response_text})
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
                    raw_text = extract_text_from_pdf(pdf_docs)
                    st.session_state.text_chunks = get_text_chunks(raw_text)
                    embedding_function = SentenceTransformerEmbeddings()
                    st.session_state.embeddings = embedding_function.embed_documents(st.session_state.text_chunks)
                    st.session_state.faiss_index, st.session_state.documents = create_faiss_index(
                        st.session_state.embeddings, st.session_state.text_chunks
                    )
                    st.session_state.documents_processed = True
                    st.success(f"✔️ {len(pdf_docs)} PDFs processados com sucesso! Agora você pode fazer perguntas.")
            except Exception as e:
                logger.exception("Erro ao processar PDFs: %s", e)
                st.error(f"Erro ao processar PDFs: {e}")
        else:
            st.warning("Carregue pelo menos um arquivo PDF para processar.")

    if 'faiss_index' in st.session_state and st.session_state.faiss_index is None and st.button("Resetar e fazer upload de novos PDFs"):
        st.session_state.documents_processed = False
        st.session_state.messages = []
        st.session_state.text_chunks = []
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        st.session_state.documents = []
        st.rerun()

st.markdown("---")
st.caption("Tarefa AS05 - André")