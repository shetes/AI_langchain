import os
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting ...")
    file_path = os.path.join(os.path.dirname(__file__), "mediumblog.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()

    print("Splitting ...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings()

    print("Ingesting to Pinecone ...")
    PineconeVectorStore.from_documents(
        documents=texts, 
        embedding=embeddings, 
        index_name=os.environ.get("INDEX_NAME")
    )
    print("Finished!")