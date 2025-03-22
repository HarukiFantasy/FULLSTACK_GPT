import json
import random
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever


# ---------------- Function Calling Schema ---------------- 

#  호출되진 않지만, 함수 이름과 파라미터 구조를 명시적으로 보여줌
def create_quiz(questions: list):
    return {"questions": questions}

# create_quiz객체 안에 questions속성 - questions는 Q & A의 속성을 가진 객체들의 배열
function = {
        "name": "create_quiz",
        "description": "Generates a multiple-choice quiz.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {  
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {  
                                "type": "string",
                                "description": "The quiz question."
                            },
                            "answers": {  
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {  
                                            "type": "string",
                                            "description": "One of the multiple-choice answers."
                                        },
                                        "correct": {  
                                            "type": "boolean",
                                            "description": "Indicates whether this answer is correct."
                                        }
                                    },
                                    "required": ["answer", "correct"]  
                                },
                            },
                            "difficulty": {  
                                "type": "string",
                                "enum": ["Easy", "Hard"], 
                                "description": "Difficulty level of the question."
                            },
                        },
                        "required": ["question", "answers", "difficulty"]  
                    }
                }
            },
            "required": ["questions"]  
        }
    }


# ------------------ 기능 구현 (docs 처리) ------------------ 

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# ------------------ Streamlit Setup ------------------ 

st.set_page_config(page_title="QuizGPT", page_icon="❓")
st.title("QuizGPT")

topic=None
with st.sidebar:
    openai_api_key = st.text_input("🔑 OpenAI API 키를 입력하세요:", type="password")
    docs = None
    if openai_api_key:
        choice = st.selectbox(
            "Choose the data you want to use.",
            ("File", "Wikipedia Article"),
            index=None
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
            
        if choice == "Wikipedia Article":
            topic = st.text_input("Search Wikipedia...", placeholder="What you want to learn?")
            if topic:
                docs = wiki_search(topic)
        if docs:
            difficulty = st.selectbox(
                "Select Difficulty Level",
                ("Easy", "Hard"),
                index=None
        )

    st.markdown(
        """
        <a href="https://github.com/HarukiFantasy/FullStackGPT" target="_blank" style="color: gray; text-decoration: none;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20">
            View on GitHub
        </a>
        """,
        unsafe_allow_html=True
    )


# ------------------ LLM & 프롬프트 설정 ------------------

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=openai_api_key,
).bind(
    function_call={"name" :"create_quiz"},
    functions=[function]
)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are an AI assistant that generates multiple-choice quiz questions. 
    Based on the given text, create at least 5 questions with 4 answer choices each.
    
    Format:
    - One correct answer per question.
    - Use the function `create_quiz` to return the questions.
    - Each question should have a difficulty level: "Easy" or "Hard"
    - **Ensure that the question itself changes depending on the difficulty level:**
    - ** If "Hard" is choosen; then show the questions with level "Hard" only. **
    - ** If "Easy" is choosen; then show the questions with level "Easy" only. **

    Type of question : 
    - **Easy:** Focus on factual recall, definitions, or simple concepts about novel and author
    - **Hard:** Ask about details of story.
    
    Context: {context}
    Difficulty Level: {difficulty}
"""
        )
    ]
)

# ------------------ 기능 구현 (quiz chain 실행) ------------------ 

def shuffle_answers(question):
    answers = question["answers"]
    random.shuffle(answers)
    return {**question, "answers": answers}

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, difficulty, topic):  
    chain = questions_prompt | llm
    response = chain.invoke({
        "context": format_docs(_docs),  
        "difficulty": difficulty  
    })
    if response.additional_kwargs.get("function_call"):
        function_name = response.additional_kwargs["function_call"]["name"]
        function_args = json.loads(response.additional_kwargs["function_call"]["arguments"])

        if "questions" in function_args:   # LLM은 가장 가능성 높은 토큰부터 순차 생성 -> 정답 섞기 
            function_args["questions"] = [shuffle_answers(q) for q in function_args["questions"]]

        if function_name == "create_quiz":
            return function_args  # 퀴즈 JSON 반환
    
    st.error("Function Calling failed. Please try again.")
    return {"questions": []}



# ------------------  메인 처리 흐름 ------------------ 

if not openai_api_key:
    st.markdown(
        """
    Welcome to QuizGPT.
    
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
    
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
    st.info("API key has not been provided.")
    st.stop()
else:
    st.session_state["openai_api_key"] = openai_api_key
    if not docs:
        st.info("Please upload file or input the topic for searching in Wikipedia ")
        st.stop()
    else:
        if not difficulty:
            st.info("Please select difficulty first to see the questions")
            st.stop()
        else:
            response = run_quiz_chain(docs, difficulty, topic)
            with st.form("questions_form"):
                for question in response["questions"]:
                    st.write(f"**{question['question']}** *(Difficulty: {question['difficulty']})*") 
                    value = st.radio("Select an option",
                            [answer["answer"] for answer in question["answers"]],
                            index=None,
                            key=f"radio_{question['question']}"
                            ) 
                    if ({"answer":value, "correct":True} in question["answers"]):
                        st.success("✅ Correct!")
                    elif value is not None:
                        st.error("❌ Wrong")
                submit_button = st.form_submit_button("Submit")
            
            if submit_button:
                # - 사용자가 선택한 값(`user_choice`)을 해당 문제의 정답(`correct_option`)과 비교해서 하나라도 틀리면 `all_correct = False`로 바꿈
                all_correct = True
                for question in response["questions"]:
                    correct_option = None
                    for answer in question["answers"]:
                        if answer["correct"]:
                            correct_option = answer["answer"]
                            break
                    user_choice = st.session_state.get(f"radio_{question['question']}")
                    if user_choice != correct_option:
                        all_correct = False
                
                if all_correct:
                    st.success("🎉  Congratulations, all answers are correct!")
                    st.balloons()
                else:
                    st.error("Not all answers are correct. Please retake the test.")
                    if st.button("🔁 Retake test"):
                        # 각 문제에 대해 세션에 저장된 값을 제거하여 선택 초기화
                        for question in response["questions"]:
                            if f"radio_{question['question']}" in st.session_state:
                                del st.session_state[f"radio_{question['question']}"]
                        st.experimental_rerun()

