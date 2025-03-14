from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from typing import Any, Dict, List
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os, requests


load_dotenv()

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
OpenAI_redirectURI = "https://chat.openai.com/aip/g-8698cc471b0ecc15590331f2ecd3867280888baf/oauth/callback"
Cloudfared_uri = "https://hc-beaver-prospect-bind.trycloudflare.com"
Cloudfared_callback_url = f"{Cloudfared_uri}/auth/callback"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "recipes"
embeddings = OpenAIEmbeddings()
vectore_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

app = FastAPI(
    title="ChefGPT. The best provider of Indian Recipes in the world",
    description="Give ChefGPT the name of an ingredient and it will give you multiple recipes to use that ingredient on in return.",
    servers=[
        {"url":Cloudfared_uri}
    ]
)

class Document(BaseModel):
    page_content: str

@app.get("/", response_class=JSONResponse)
def root():
    return {"message": "Welcome to the Cooking recipes API!"}

@app.get("/recipes", response_model=List[Document])
async def get_receipt(request: Request, ingredient: str):
    docs = vectore_store.similarity_search(ingredient)
    return docs

@app.get("/auth")
def github_login(state: str):
    """사용자를 GitHub OAuth 인증 페이지로 리디렉션"""
    github_auth_url = (
        f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&redirect_uri={Cloudfared_callback_url}&scope=read:user&state={state}"
    )
    return RedirectResponse(github_auth_url)

@app.get("/auth/callback")
def github_callback(request: Request):
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code:
        return {"error": "No code provided"}

    token_response = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        json={
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code,
            "redirect_uri": Cloudfared_callback_url,
        },
    )

    token_data = token_response.json()
    access_token = token_data.get("access_token")
    # 2. 유저 정보 요청
    user_response = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    user_data = user_response.json()
    github_user = user_data.get('login')
    # 3. CustomGPT로 리디렉션
    redirect_url = f"{OpenAI_redirectURI}?code={code}&state={state}"
    return RedirectResponse(redirect_url)


# OAuth 토큰 요청 처리
@app.post("/token", include_in_schema=False,)
async def handle_oauth_token(code: str = Form(...)):
    token_url = "https://github.com/login/oauth/access_token"
    headers = {"Accept": "application/json"}
    payload = {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "code": code,
        "redirect_uri": OpenAI_redirectURI
    }
    
    response = requests.post(token_url, headers=headers, data=payload)
    data = response.json()

    user_data = response.json()
    print(user_data)

    user_id = str(user_data.get("id"))
    user_email = user_data.get("email")  # GitHub에서 유저 이메일 가져오기
    user_name = user_data.get("login")  # GitHub 유저명 가져오기

    return {"access_token": user_id} 