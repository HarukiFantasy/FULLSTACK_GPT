import os, hashlib, requests
from bs4 import BeautifulSoup
from langchain.document_loaders import SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate
import streamlit as st

# ---------- Streamlit UI 설정 ----------

st.set_page_config(page_title="SiteGPT", page_icon="🖥️")
st.markdown("""
# SiteGPT
Ask questions about the content of a website.
Start by writing the URL of the website on the sidebar.
""")

with st.sidebar:
    openai_api_key = st.text_input("🔑 Please enter your OpenAI API key:", type="password")
    url, keyword = None, None
    if openai_api_key:
        url = st.text_input("Write down a URL", placeholder="https://example.com/sitemap.xml")
        if url:
            keyword = st.text_input("🔍 Enter the product name or keyword you want to search:", placeholder="ex: gateway, vectorize, workersai")

    st.markdown("""
    <a href="https://github.com/HarukiFantasy/FULLSTACK_GPT" target="_blank" style="color: gray; text-decoration: none;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20">
        View on GitHub
    </a>
    """, unsafe_allow_html=True)


# ---------- 기능 구현 (메세지) ----------

if "sitegpt_messages" not in st.session_state:
    st.session_state["sitegpt_messages"] = []

def save_message(message, role):
    st.session_state["sitegpt_messages"].append({"sitegpt_message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for msg in st.session_state["sitegpt_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["sitegpt_message"])


# ---------- 기능구현 (사이트 파싱) ----------

def filter_urls_from_sitemap(sitemap_url, keyword):
    response = requests.get(sitemap_url)
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(response.content, "xml")
    loader = SitemapLoader(web_path=sitemap_url)
    sitemap_data = loader.parse_sitemap(soup)
    filtered_urls = [item["loc"] for item in sitemap_data if keyword.lower() in item["loc"].lower()]
    return filtered_urls

Html2Text_transformer = Html2TextTransformer()

@st.cache_resource(show_spinner="Loading filtered product pages...")
def load_filtered_product_pages(urls):
    url_hash = hashlib.md5(("".join(urls)).encode("utf-8")).hexdigest()
    persist_directory = f"./.cache/site_files/faiss_{url_hash}"

    if os.path.exists(persist_directory):
        vector_store = FAISS.load_local(persist_directory, OpenAIEmbeddings())
        return vector_store.as_retriever()

    progress_bar = st.progress(0, text="📄 Loading documents...")
    status_text = st.empty()
    documents = []
    total = len(urls)

    for i, url in enumerate(urls):
        status_text.markdown(f"🔗 **Processing:** {url}")
        try:
            # Playwright 제거 → requests + BeautifulSoup 방식으로 대체
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            html_text = soup.get_text(separator=" ", strip=True)

            from langchain.schema import Document
            doc = Document(page_content=html_text, metadata={"source": url})
            transformed = Html2Text_transformer.transform_documents([doc])
            documents.extend(transformed)

        except Exception as e:
            st.warning(f"⚠️ Failed to load {url}: {e}")
        progress_bar.progress((i + 1) / total, text=f"📄 Loaded {i+1} of {total} documents")

    progress_bar.empty()

    if not documents:
        st.info("❌ Unable to load the page.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=800, chunk_overlap=100
    )
    split_docs = splitter.split_documents(documents)
    vector_store = FAISS.from_documents(split_docs, OpenAIEmbeddings(model="text-embedding-3-small"))
    vector_store.save_local(persist_directory)
    return vector_store.as_retriever()


# ---------- 기능 구현 (체인 구성) ----------

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm_for_internal
    results = []

    for doc in docs:
        answer_raw = answers_chain.invoke({
            "question": question,
            "context": doc.page_content,
        }).content

        results.append({
            "answer": answer_raw,
            "source": doc.metadata["source"],
            })
    return {"question": question, "answers": results}

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\n"
        for answer in answers
    )
    with st.chat_message("ai"):
        answer = choose_chain.invoke({"question": question, "answers": condensed})
    return answer


# ---------- 콜백 핸들러 ----------

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

# ---------- LLM 및 프롬프트 설정 ----------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=openai_api_key,
    streaming=True,
    callbacks=[ChatCallbackHander()]
)

llm_for_internal = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=openai_api_key,
    streaming=False
)

answers_prompt = ChatPromptTemplate.from_template("""
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
""")

choose_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Use ONLY the following pre-existing answers to answer the user's question.
    Use the answers that have the highest score (more helpful) and favor the most recent ones.

    Cite sources and return the sources of the answers as they are, do not change them.
    {answers}
    """),
    ("human", "{question}"),
])

# ---------- 메인 흐름 ----------

if not openai_api_key:
    st.info("API key has not been provided.")
    st.stop()

if url and keyword:
    filtered_urls = filter_urls_from_sitemap(url, keyword)
    if not filtered_urls:
        st.info("❌ No links containing the keyword were found")
        st.stop()

    st.markdown("### 🔗 Select the link:")
    selected_links = st.multiselect(
        "✅ Select the links to ask about. Subpages of selected links are also included.",
        options=filtered_urls,
        default=[],
        placeholder="Select the links"
    )

    if selected_links:
        expanded_links = []
        for selected in selected_links:
            children = [link for link in filtered_urls if link.startswith(selected)]
            expanded_links.extend(children)
        expanded_links = list(set(expanded_links))
        if st.button("🚀 Start asking questions with the selected links"):
            st.session_state["selected_links"] = expanded_links
            st.rerun()

if "selected_links" in st.session_state:
    selected_links = st.session_state["selected_links"]
    st.success(f"🔍 Number of selected links : {len(selected_links)}")
    retriever = load_filtered_product_pages(selected_links)

    if retriever:
        paint_history()
        user_input = st.chat_input("What would you like to know in the content of selected link?")
        if user_input:
            send_message(user_input, "human")
            chain = (
                {
                    "docs": RunnableLambda(lambda q: retriever.invoke(q["question"])),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke({"question": user_input})
