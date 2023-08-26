from dotenv import load_dotenv

load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Loader
loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()  # page별로 split

# Split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,  # 자르는 사이즈
    chunk_overlap=20,  # 겹치는 글자 수를 정해줌
    length_function=len,
    is_separator_regex=False,  # 정규표현식으로 split할 때 사용
)

texts = text_splitter.split_documents(pages)

# Embedding
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
