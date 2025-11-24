import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print(" Retrieving ...")

    # Initialize the OpenAIEmbeddings object, which is used to generate vector representations (embeddings)
    # of text for retrieval tasks.
    embeddings = OpenAIEmbeddings()
    
    # Initialize the ChatOpenAI object, representing the language model that will be used to generate answers.
    llm = ChatOpenAI()

    # Define the user's query to be answered using retrieved documents and an LLM.
    query = "What is Pinecone in machine learning?"

    # Set up a PineconeVectorStore for performing vector similarity searches.
    # The vector store is initialized with the index name and the embedding function.
    vectorstore = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
    )

    # Pull a prompt template for a retrieval-based question-answering chat chain from the LangChain hub.
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a document-combination chain. This sets up how retrieved documents and the prompt are
    # fed to the language model.
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # Create a retrieval chain that will:
    # 1. Retrieve documents using the vector store based on the user's query.
    # 2. Pass those documents and the query through the document-combination chain to generate an answer.
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})

    print(result.get("answer"))

    template = """ Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Here we're building a Retrieval-Augmented Generation (RAG) pipeline:
    # 1. We create a dictionary where:
    #    - The "context" key retrieves relevant documents for the query, then formats them.
    #    - The "question" key simply passes through the user query as-is.
    # 2. This dictionary is run through a custom prompt template, inserting the context and question into the prompt.
    # 3. The final prompt is sent to the language model (llm) to generate the answer.

    dict = {
            "context": vectorstore.as_retriever() | format_docs,   # Retrieves and formats context docs
            "question": RunnablePassthrough()                      # Forwards the question unchanged
        }

    rag_chain = (
        dict
        | custom_rag_prompt                                         # Assembles the prompt
        | llm                                                       # Gets the answer from the LLM
    )

    # We invoke the rag_chain with our query, which triggers all the above steps and returns the LLM's answer.
    result2 = rag_chain.invoke(query)
    print(result2)