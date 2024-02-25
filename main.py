import os

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

# load env from local .env file
load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main() -> None:
    # load .txt file
    loader = TextLoader("media/example.txt")
    documents = loader.load()

    # split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20, length_function=len, is_separator_regex=False
    )
    splits = text_splitter.split_documents(documents)

    # create embeddings
    embeddings = OpenAIEmbeddings()

    vectorstore = None
    if os.path.exists("vectordb/"):
        vectorstore = FAISS.load_local(f"vectordb/", embeddings)
    else:
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local("vectordb/")

    retriever = vectorstore.as_retriever()

    query = {"context": retriever | format_docs, "question": RunnablePassthrough()}
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
    oputput_parser = StrOutputParser()

    # create rag chain
    chain = query | prompt | llm | oputput_parser

    while True:
        question = input("Enter your question: ")

        print("\nProcessing your query...\n")

        response = chain.invoke(question)

        print("ChatGPT ==>", response)
        print()


if __name__ == "__main__":
    main()
