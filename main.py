__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

# from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

button(username="kanique", floating=True, width=221)

# load_dotenv()

# Title
st.title("ChatPDF")
st.write("---")

# OpenAI KEY 입력받기
openai_key = st.text_input("OPEN_AI_API_KEY", type="password")

# Upload File
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=["pdf"])
st.write("---")


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()  # page별로 split
    return pages


# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=300,  # 자르는 사이즈
        chunk_overlap=20,  # 겹치는 글자 수를 정해줌
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Question
    st.header("PDF에게 질문해 보세요!!")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        with st.spinner("답변 작성 중..."):
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key
            )
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
