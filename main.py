import os

from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_core.tools import tool
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

### data in st sessions
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
    ## load documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ## split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents=docs)

    ### embedding and Vector DB
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = InMemoryVectorStore.from_documents(embedding=embedding, documents=docs)

    ## create a agent - tool, llm, prompt
    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query: str):
        """
        Retrieve documents relevant to a query from the knowledge base.
        """
        docs = vector_db.similarity_search(query=query, k=3)
        context = ""
        for doc in docs:
            context += doc.page_content + "\n\n"
        return context

    system_prompt = """
    You are a helpful assistant that answers questions
    using the uploaded documents.
    For questions about information contained in the
    knowledge base, ALWAYS call the retrieve_context tool.
    Base your answers on the retrieved context.
    If the retrieved context does not contain the answer,
    clearly tell the user that the information is not available.
    Do not invent medical information.
    """

    memory = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


## upload ui
st.header("End to End Chatbot")

## Take files inputs from user and save it
## Pass the files to the agent and save to session
## Maintain a boolean state for file upload, and render ui based on it
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./docs_files/"
            os.makedirs(path, exist_ok=True)  # added: make sure the upload folder exists
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()

## chat ui
## User already uploaded the file and agent is created with it
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything to related to uploaded documents...")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        response = st.session_state.agent.invoke({
            "messages": [{"role": "user", "content": query}]
        }, {"configurable": {"thread_id": "session_123"}})

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({
            "role": "ai",
            "content": answer
        })
