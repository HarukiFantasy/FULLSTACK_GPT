from langchain.document_loaders import AsyncChromiumLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 문서들 로딩
loader = AsyncChromiumLoader(["https://example.com/page1", "https://example.com/page2"])
docs = loader.load()

# 문서 스코어링: 길이 기준으로 내림차순 정렬
sorted_docs = sorted(docs, key=lambda d: len(d.page_content), reverse=True)

# 가장 "중요해 보이는" 문서 1개 선택
top_doc = sorted_docs[0:1]

# 선택한 문서 쪼개기
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=800, chunk_overlap=100
)
split_docs = splitter.split_documents(top_doc)

# 청크 중 맨 앞 하나만 사용
limited_docs = split_docs[:1]

# 벡터 저장소 생성
vector_store = FAISS.from_documents(
    limited_docs,
    OpenAIEmbeddings(model="text-embedding-3-small")
)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})
