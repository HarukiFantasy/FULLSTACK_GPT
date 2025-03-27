import streamlit as st
import openai as client
import json, yfinance, time
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from typing_extensions import override

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="InvestorGPT", page_icon="💹")
st.title("InvestorGPT")
st.markdown("Welcome to InvestorGPT!")

with st.sidebar:
    openai_api_key = st.text_input("🔑 Please enter your OpenAI API Key to proceed:", type="password")
    st.markdown("""
        <a href="https://github.com/HarukiFantasy/FullStackGPT" target="_blank" style="color: gray; text-decoration: none;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20">
            View on GitHub
        </a>
    """, unsafe_allow_html=True)

if not openai_api_key:
    st.info("API key has not been provided.")
    st.stop()

client.api_key = openai_api_key
assistant_id = "asst_IYOQtkSi1ftVTrQhznFTGFdi" 

# ------------------ 툴 함수 정의 ------------------
def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(f"Ticker symbol of {inputs['company_name']}")

def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())

def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())

def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())

functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}

# ------------------ 메시지 관련 ------------------

# 🎯 대화 기록 저장 (최초 실행 시 초기화)
if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id
    st.session_state.messages = []  # 대화 기록 저장

# 🎯 실행 상태 확인
def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)

# 🎯 최종 메시지 가져오기 및 UI 업데이트
def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()  # 최신 메시지를 위로 정렬
    
    # 🔥 Streamlit UI에 대화 업데이트
    st.session_state.messages = messages

# 🎯 requires_action 상태일 때 실행할 함수
def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    if run.status == "requires_action":
        for action in run.required_action.submit_tool_outputs.tool_calls:
            action_id = action.id
            function = action.function
            print(f"Calling function: {function.name} with arg {function.arguments}")
            outputs.append(
                {
                    "output": functions_map[function.name](json.loads(function.arguments)),
                    "tool_call_id": action_id,
                }
            )
    return outputs

# 🎯 Tool Outputs 제출
def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    if outputs:
        client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id, thread_id=thread_id, tool_outputs=outputs
        )

# 🎯 메시지 전송 및 실행 (Streamlit UI 포함)
def send_message(user_input):
    thread_id = st.session_state.thread_id  # 기존 쓰레드 사용

    # 🔹 기존 쓰레드에 메시지 추가
    client.beta.threads.messages.create(
        thread_id=thread_id, 
        role="user", 
        content=user_input
    )

    # 🔹 실행 시작 (get_run)
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    # 🔹 실행이 완료될 때까지 기다림
    while True:
        run_status = get_run(run.id, thread_id)
        print(f"🔄 현재 실행 상태: {run_status.status}")

        if run_status.status == "requires_action":
            print("⚡ Function Calling detected, retrieving tool outputs...")
            submit_tool_outputs(run.id, thread_id)  # 🔥 requires_action 시 outputs 제출
        elif run_status.status in ["completed", "failed", "cancelled"]:
            print(f"✅ 실행 완료! 상태: {run_status.status}")
            break
        
        time.sleep(2)  # 2초 대기 후 다시 확인

    # 🔹 최종 메시지 가져와 Streamlit UI에 업데이트
    get_messages(thread_id)

user_input = st.chat_input("💬 Type your message and press Enter...")
if user_input:
    send_message(user_input) 

# 🎯 Streamlit - 대화 기록 UI 표시
for message in st.session_state.messages:  
    with st.chat_message(message.role):
        st.markdown(message.content[0].text.value)