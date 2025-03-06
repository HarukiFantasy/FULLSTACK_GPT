import os
import hashlib
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT
    Ask questions about the content of a website.
    Start by writing the URL of the website on the sidebar.
"""
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["message"])

def get_chat_history():
    return "\n".join(f"{msg['role']}: {msg['message']}" 
                    for msg in st.session_state["messages"])

# API 키 입력 필드
with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com/sitemap.xml")
    openai_api_key = st.text_input("🔑 OpenAI API 키를 입력하세요:", type="password")
    st.markdown(
    """
    <a href="https://github.com/HarukiFantasy/Fullstack-gpt" target="_blank" style="color: gray; text-decoration: none;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20">
        View on GitHub
    </a>
    """,
    unsafe_allow_html=True
)
    
if not openai_api_key:
    st.info("API key has not been provided.")
    st.stop()

# API 키를 환경 변수에 설정
os.environ["OPENAI_API_KEY"] = openai_api_key

llm = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key)

# 프롬프트 수정
answers_prompt = ChatPromptTemplate.from_template(
    """
    Chat History:
    {chat_history}

    Website Content:
    {context}

    Using the above information, answer the user's question.
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    Question: {question}
    """
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    chat_history = get_chat_history()
    
    answers_chain = answers_prompt | llm
    
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question, 
                        "context": doc.page_content,
                        "chat_history": chat_history
                    }
                ).content,
                "source": doc.metadata.get("source", "Unknown"),
                "date": doc.metadata.get("lastmod", "Unknown"),
            } 
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    
    choose_chain = choose_prompt | llm
    
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    
    return choose_chain.invoke({"question": question, "answers": condensed}).content

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    persist_directory = f"./.cache/site_files/faiss_{url_hash}"
    
    if os.path.exists(persist_directory):
        vector_store = FAISS.load_local(persist_directory, OpenAIEmbeddings())
        return vector_store.as_retriever()
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    loader = SitemapLoader(url, parsing_function=parse_page)
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)

    # 병렬 로딩을 적용할 필요 없음 (SitemapLoader가 자동으로 처리)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    vector_store.save_local(persist_directory)

    return vector_store.as_retriever()

if url:
    if not url.endswith(".xml"):
        with st.sidebar:
            st.error("Please write down a valid Sitemap URL.")
    else:
        retriever = load_website(url)

        paint_history()
        user_input = st.chat_input("Ask a question about the website.")
        
        if user_input:
            send_message(user_input, "human")
            chain = (
                {
                    "docs": RunnableLambda(lambda q: retriever.invoke(q)),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            
            result = chain.invoke({"question": user_input})
            answer = result.replace("$", "\$")
            send_message(answer, "ai")