from crewai_tools import GithubSearchTool
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["MISTRAL_API_KEY"]=os.getenv("MISTRAL_API_KEY")

from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"]=os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

loader = GithubFileLoader(
    repo="JISHNU-SUNEESH/Generative_AI",  # the repo name
    branch="master",  # the branch name
    ccess_token=os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
    github_api_url=" https://api.github.com",
    file_filter=lambda file_path: file_path.endswith(
        ".py"
    ),  # load all markdowns files.
)
documents = loader.load()
spliter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
splits=spliter.split_documents(documents)

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

llm = ChatMistralAI(model="mistral-large-latest")
embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = Chroma(embedding_function=embeddings)
vector_store.add_documents(documents=splits)
retriever=vector_store.as_retriever()
system_prompt=(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}"),

    ]
)

qa_chain=create_stuff_documents_chain(llm,qa_prompt)
rag_chain=create_retrieval_chain(retriever,qa_chain)

result=rag_chain.invoke({"input":"how many documets are there?"})
print(result)