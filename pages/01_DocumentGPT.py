import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

# ----- Streamlit UI 설정 -----

st.set_page_config(page_title="DocumentGPT", page_icon="📑")
st.title("DocumentGPT")

st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!
    Provide API Key and Upload your files on the sidebar.
    """
    )

with st.sidebar:
    openai_api_key = st.text_input("🔑 OpenAI API 키를 입력하세요:", type="password")
    file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["pdf", "txt", "docx"])
    st.markdown(
    """
    <a href="https://github.com/HarukiFantasy/FULLSTACK_GPT" target="_blank" style="color: gray; text-decoration: none;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20">
        View on GitHub
    </a>
    """,
    unsafe_allow_html=True
)
    
if not openai_api_key:
    st.info("API key has not been provided.")
    st.stop()
if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key


# ----- Callback for streaming LLM responses -----

class ChatCallbackHander(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None  

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# ----- LLM 생성 -----

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    openai_api_key=openai_api_key,
    callbacks=[ChatCallbackHander()]
)

# ----- 임베딩 처리 & 파일 처리 -----

@st.cache_resource(show_spinner="Embedding file...") 
def embed_file(file, openai_api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", 
        chunk_size=600,
        chunk_overlap=100
    )      
    docs = UnstructuredFileLoader(file_path).load_and_split(text_splitter=splitter)
    embeddings = CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(openai_api_key=openai_api_key), cache_dir
    )
    return FAISS.from_documents(docs, embeddings).as_retriever()

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# ----- 메모리 설정 : 대화내역을 llm 에 전달하기 위함 -----

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
memory = st.session_state["memory"]  

def load_memory(_):
    return memory.load_memory_variables({}).get("chat_history", [])


# ----- 메세지 설정 및 기능 구현 -----

st.session_state.setdefault("messages", [])

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


# ----- 프롬프트 구성 -----

prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    Given the following extracted parts of a long document and a question, create a final answer.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    -------
    Context: {context}
    """),
    MessagesPlaceholder(variable_name="chat_history"),  
    ("human", "{question}")
])

# ----- 메인 처리 흐름 ----- 

if file:
    if "openai_api_key" not in st.session_state or not st.session_state["openai_api_key"]:
        st.warning("⚠️ API Key is required!")
        st.stop()
    retriever = embed_file(file, openai_api_key)

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    
    message = st.chat_input("Ask anything about your file...") 

    if message:
        send_message(message, "human")

        chain = {
            "chat_history": RunnableLambda(load_memory), 
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm

        with st.chat_message("ai"):
            response = chain.invoke(message)

        memory.save_context(
            inputs={"input": message}, 
            outputs={"output": response.content}
        )