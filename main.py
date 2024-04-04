import os

from langchain import hub

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv


load_dotenv(".env")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain():
    data = TextLoader("example.txt")
    docs = data.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=100, chunk_overlap=20, length_function=len
    )
    text_splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_splits, embeddings)
    retriever = vectorstore.as_retriever()

    query = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.8,
    )
    rag_prompt = hub.pull("rlm/rag-prompt")
    oputput_parser = StrOutputParser()

    chain = query | rag_prompt | llm | oputput_parser

    return chain


def main() -> None:
    pass


if __name__ == "__main__":
    main()
