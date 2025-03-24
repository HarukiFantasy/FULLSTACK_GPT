import streamlit as st
import time, os, requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import SystemMessage


# ------------------ Streamlit Setup ------------------ 
st.set_page_config(page_title="InvestorGPT", page_icon="💹")
st.title("InvestorGPT")
st.markdown(
    """
    Welcome to InvestorGPT!
    """)


with st.sidebar:
    openai_api_key = st.text_input("🔑 OpenAI API 키를 입력하세요:", type="password")

if not openai_api_key:
    st.info("API key has not been provided.")
    st.stop()

llm = ChatOpenAI(
    model_name="gpt-4o-mini", 
    temperature = 0.1,
    streaming=True,
    openai_api_key=openai_api_key
    )

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")

class StockMarketSymbolSearchTool(BaseTool):
    name: str = "StockMarketSymnbolSearchTool"
    description: str = """
    Use this tool to find the stock market symbol for a compnay. 
    It takes a query as an argument.
    Example query: Storck Market Symbol for Apple Company
    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        result = ddg.run(query)
        time.sleep(3)  # Adding delay to avoid rate limiting
        return result

class CompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(description="Stock symbol of the company. Example: AAPL, TSLA")

class CompnayOverviewTool(BaseTool):
    name: str = "CompnayOverviewTool"
    description: str = """ 
    Use this to get an overview of the financials of the company. 
    You should enter a stock symbol. 
    """
    args_schema: Type [CompanyOverviewArgsSchema]=CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()
    

class CompnayIncomeStatementTool(BaseTool):
    name: str = "CompnayIncomeStatementTool"
    description: str = """ 
    Use this to get an income statement of the company. 
    You should enter a stock symbol. 
    """
    args_schema: Type [CompanyOverviewArgsSchema]=CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()["annualReports"]
    

class CompnayStockPerformanceTool(BaseTool):
    name: str = "CompnayStockPerformanceTool"
    description: str = """ 
    Use this to get a weely performance of the company. 
    You should enter a stock symbol. 
    """
    args_schema: Type [CompanyOverviewArgsSchema]=CompanyOverviewArgsSchema
    
    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}")
        response = r.json()
        return list(response["Weekly Time Series"].items())[:5]  
        # 딕셔너리를 리스트로 만든뒤 일부분 5까지 (5주치) 자료 가져옴
        # { "Weekly Time Series" : {"x1":1, "x2":2, "x3":3} } -> dic_items([ ("x1",1), ("x2",2), ("x3",3) ])
    
agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompnayOverviewTool(),
        CompnayIncomeStatementTool(),
        CompnayStockPerformanceTool()
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a hedge fund manager.
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
            Consider the performance of a stock, the company overview and the income statement.
            Be assertive in your judgement and recommend the stock or advise the user against it.
        """)
    } # 설정하지 않으면 기본값인 "You are a helpful AI assistant" 정의로 처리된다
)

company = st.text_input("Write the name of the company you are interested in")

if company:
    result = agent.invoke(company)
    st.write(result["output"].replace("$", "\$"))