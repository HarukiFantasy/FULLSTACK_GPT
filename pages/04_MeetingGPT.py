from langchain.storage import LocalFileStore
import streamlit as st
import subprocess, math, glob, openai, os, time
from pydub import AudioSegment
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings


llm = ChatOpenAI(model="gpt-3.5-turbo-1106" ,temperature=0.1,)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


@st.cache_data()
def embed_file(file_path):
    file_name = os.path.basename(file_path)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# ✅ 비디오에서 오디오만 추출하기
# terminal command : ffmpeg -i 파일명.mp4 -vn 파일명.mp3 / -y : Always Yes로 설정
@st.cache_data()
def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4", "mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


# ✅ 오디오 길이 계산하고 chunk_size(분)단위로 잘라내고 저장
# track.duration_seconds 오디오 길이 확인 (밀리세컨 단위)
@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"./{chunks_folder}/chunk_{i}.mp3",
            format="mp3",
        )


# ✅ 파일 경로를 하나씩 불러와 텍스트화 후 저장 (a : append 모드)
@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    files = glob.glob(f"{chunk_folder}/*.mp3") # glob 파일 경로를 리스트로 만들기
    files.sort() # 파일 정렬
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1", audio_file
            ) # transcript 객체 :  dic타입 {"text" : "스크립트"}
            text_file.write(transcript["text"])


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    openai_api_key = st.text_input("🔑 OpenAI API 키를 입력하세요:", type="password")
    st.markdown(
    """
    <a href="https://github.com/HarukiFantasy/Fullstack-gpt" target="_blank" style="color: gray; text-decoration: none;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20">
        View on GitHub
    </a>
    """)

    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )
    if video:
        # status_container = st.empty() 
        chunks_folder = "./.cache/chunks"
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        #  ✅ 비디오에서 오디오로 변환 경로/텍스트 변환 경로 설정 
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        st.session_state["transcript_path"] = transcript_path

        with st.status("Loading video...") as status:
            # ✅ 파일 읽고, 오디오 추출, 분할
            with open(video_path, "wb") as f:
                f.write(video_content)
            status.update(label="Extracting audio...", state="running")
            extract_audio_from_video(video_path)
            status.update(label="Cutting audio segments...", state="running")
            cut_audio_in_chunks(audio_path, 3, chunks_folder)
            status.update(label="Transcribing audio...", state="running")
            transcribe_chunks(chunks_folder, transcript_path)
            status.update(label="Complete!", state="complete")
            time.sleep(2) 
        # status_container.empty()

if not openai_api_key:
    st.info("API key has not been provided.")
    st.stop()

if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key


transcript_tab, summary_tab, qa_tab = st.tabs(
    ["Transcript", "Summary", "Q&A"]
)

@st.cache_data()
def load_transcript(transcript_path):
    with open(transcript_path, "r") as file:
        return file.read()

with transcript_tab:
    if "transcript_path" in st.session_state:
        transcript_text = load_transcript(st.session_state["transcript_path"])
        st.write(transcript_text)
    else:
        st.write("No transcript available. Please upload a video.")


@st.cache_data(show_spinner=False)
def generate_summary(transcript_path):
    loader = TextLoader(transcript_path)
    docs = loader.load_and_split(text_splitter=splitter)
    first_summary_prompt = ChatPromptTemplate.from_template(
        """
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:                
    """
    )
    first_summary_chain = first_summary_prompt | llm | StrOutputParser()
    first_summary = first_summary_chain.invoke(
        {"text": docs[0].page_content},
    )

    refine_prompt = ChatPromptTemplate.from_template(
        """
        Your job is to produce a final summary.
        We have provided an existing summary up to a certain point: {existing_summary}
        We have the opportunity to refine the existing summary (only if needed) with some more context below.
        ------------
        {context}
        ------------
        Given the new context, refine the original summary.
        If the context isn't useful, RETURN the original summary.
        """
    )
    refine_chain = refine_prompt | llm | StrOutputParser()
    with st.status("Summarizing...") as status:
        for i, doc in enumerate(docs[1:]):  # 인덱스와 문서를 함께 가져옴
            status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
            
            summary = refine_chain.invoke(
                {
                    "existing_summary": first_summary,
                    "context": doc.page_content,
                }
            )
    return summary

with summary_tab:
    if "transcript_path" in st.session_state:
        start = st.button("Generate Summary")  # 버튼 클릭 시만 실행
        if start:
            summary = generate_summary(st.session_state["transcript_path"])
            st.write(summary)
    else:
        st.write("No transcript available. Please upload a video.")



@st.cache_data()
def get_retriever(transcript_path):
    return embed_file(transcript_path)

with qa_tab:
    if "transcript_path" in st.session_state:
        retriever = get_retriever(st.session_state["transcript_path"])

        if "responses" not in st.session_state:
            st.session_state["responses"] = []

        question = st.text_input(label="Ask about meeting!")
        
        if question:
            map_doc_prompt = ChatPromptTemplate.from_template(
                """
                Use the following context of a long document to see if any of the text is relevant to answer the {question}.
                ---
                {context}
                """
            )    
            map_doc_chain = map_doc_prompt | llm
            def map_docs(inputs):
                documents = inputs["documents"]
                question = inputs["question"]
                return "\n\n".join(
                    map_doc_chain.invoke(
                        {"context": doc.page_content, "question": question}
                    ).content
                    for doc in documents)

            map_chain = {"documents": retriever, "question": RunnablePassthrough()} | RunnableLambda(map_docs) 
            final_prompt = ChatPromptTemplate.from_template( 
                """
                Use the following context of a long document to see if any of the text is relevant to answer the {question}.
                ---
                {context}
                """
                )
            chain = {"context": map_chain, "question": RunnablePassthrough()} | final_prompt | llm | StrOutputParser()
            response = chain.invoke(question)
            st.session_state["responses"].append(response)

        def paint_history():
            for response in st.session_state["responses"]:
                st.markdown(response)

        paint_history()
    else:
        st.write("No transcript available. Please upload a video.")