import os
import streamlit as st
import PyPDF2
from io import StringIO
import docx2txt
from langchain.agents import initialize_agent
#from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

from langchain.chains import ConversationalRetrievalChain

from tools.search_ddg import get_search_ddg_tool
from tools.fetch_page import get_fetch_page_tool

CUSTOM_SYSTEM_PROMPT = """You are an assistant that conducts online research based on user requests. Using available tools, please explain the researched information.
Please don't answer based solely on what you already know. Always perform a search before providing a response.

In special cases, such as when the user specifies a page to read, there's no need to search.
Please read the provided page and answer the user's question accordingly.

If you find that there's not much information just by looking at the search results page, consider these two options and try them out.
Users usually don't ask extremely unusual questions, so you'll likely find an answer:

- Try clicking on the links of the search results to access and read the content of each page.
- Change your search query and perform a new search.

Users are extremely busy and not as free as you are.
Therefore, to save the user's effort, please provide direct answers.

BAD ANSWER EXAMPLE
- Please refer to these pages.
- You can write code referring these pages.
- Following page will be helpful.

GOOD ANSWER EXAMPLE
- This is sample code:  -- sample code here --
- The answer of you question is -- answer here --

Please make sure to list the URLs of the pages you referenced at the end of your answer. (This will allow users to verify your response.)

Please make sure to answer in the language used by the user. If the user asks in Japanese, please answer in Japanese. If the user asks in Spanish, please answer in Spanish.
But, you can go ahead and search in English, especially for programming-related questions. PLEASE MAKE SURE TO ALWAYS SEARCH IN ENGLISH FOR THOSE.
"""


def init_page():
    st.set_page_config(
        page_title="Conversational Agents",
        page_icon="🚀"
    )
    st.sidebar.title("Config")
    st.session_state["openai_api_key"] = st.sidebar.text_input(
        "OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    st.session_state["agent_type"] = st.sidebar.selectbox("Choose Agent Type", options=["Web Agent", "RAG Agent"])
    if st.session_state.agent_type == "Web Agent":
        st.header("Web Browsing Conversational Agent")
    elif st.session_state.agent_type == "RAG Agent":
        st.header("RAG Conversational Agent")
    
    
    
    # st.session_state["langsmith_api_key"] = st.sidebar.text_input(
    #     "LangSmith API Key (optional)", type="password")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the Web/Doc. How can I help you?"}
        ]
        st.session_state.costs = []


def select_model():
    models = ["GPT-4", "gpt-4-turbo", "gpt-4-vision-preview","gpt-3.5-turbo-0125", "GPT-3.5-16k",  "GPT-3.5 (not recommended)"]
    model = st.sidebar.selectbox("Choose a Model", options=models) #radio("Choose a model:", ("GPT-4", "GPT-3.5-16k",  "GPT-3.5 (not recommended)"))
    if model == "GPT-4":
        st.session_state.model_name = "gpt-4"
    elif model == "gpt-4-turbo":
        st.session_state.model_name = "gpt-4-turbo"
    elif model == "gpt-4-vision-preview":
        st.session_state.model_name = "gpt-4-vision-preview"
    elif model == "gpt-3.5-turbo-0125":
        st.session_state.model_name = "gpt-3.5-turbo-0125"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    elif model == "GPT-3.5 (not recommended)":
        st.session_state.model_name = "gpt-3.5-turbo"
    else:
        raise NotImplementedError
    
    temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    
    return ChatOpenAI(
        temperature=temp,
        openai_api_key=st.session_state["openai_api_key"],
        model_name=st.session_state.model_name,
        streaming=True
    )


def main():
    init_page()
    init_messages()
    tools = [get_search_ddg_tool(), get_fetch_page_tool()]

    # """
    # This is a sample Web Browsing Agent app that uses LangChain's `OpenAIFunctionsAgent` and Streamlit's `StreamlitCallbackHandler`. Please refer to the code for more details at https://github.com/naotaka1128/web_bowsing_agent.
    # """

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if not st.session_state["openai_api_key"]:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    else:
        llm = select_model()

    # if st.session_state["langsmith_api_key"]:
    #     os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    #     os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
    #     os.environ['LANGCHAIN_API_KEY'] = st.session_state["langsmith_api_key"]
    if st.session_state.agent_type == "Web Agent":
        st.sidebar.text_area(label = "SYSTEM PROMPT", value=CUSTOM_SYSTEM_PROMPT, height=350)
        if prompt := st.chat_input(placeholder="Who won the 1992 Cricket World Cup?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            search_agent = initialize_agent(
                agent='openai-functions',
                tools=tools,
                llm=llm,
                max_iteration=5,
                agent_kwargs={
                    "system_message":  SystemMessage(content=CUSTOM_SYSTEM_PROMPT)
                }
            )
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
    elif st.session_state.agent_type == "PDF Agent":
        @st.cache_data
        def load_docs(files):
            all_text = []
            for file_path in files:
                file_extension = os.path.splitext(file_path.name)[1]
                if file_extension == ".pdf":
                    pdf_reader = PyPDF2.PdfReader(file_path)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    all_text.append(text)
                elif file_extension == ".txt":
                    stringio = StringIO(file_path.getvalue().decode("utf-8"))
                    text = stringio.read()
                    all_text.append(text)
                elif file_extension == ".docx":
                    docx_loader = docx2txt.process(file_path) #UnstructuredWordDocumentLoader(file_path)
                    #docx_doc = docx_loader.load()
                    #st.write(docx_loader)
                    all_text.append(docx_loader)
                else:
                    st.warning('Please provide txt/pdf/docx/doc.', icon="⚠️")
            # st.write(all_text)
            return all_text 
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        st.sidebar.write("Upload PDF/TXT/DOCX/DOC Files:")
        uploaded_files = st.sidebar.file_uploader("Upload", type=["pdf", "txt", "docx", "doc"], label_visibility="collapsed", accept_multiple_files = True)
        if uploaded_files != []:
            #st.sidebar.info('Initializing Document Loading...')
            @st.cache_resource
            def retr(uploaded_files):
                documents = load_docs(uploaded_files)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                docs = text_splitter.create_documents(documents)
                embeddings = OpenAIEmbeddings()
                db = FAISS.from_documents(docs, embeddings)
                retriever = db.as_retriever(k = 1)
                st.sidebar.success("Document Loaded Successfully!") 
                return retriever, db
            retriever, db = retr(uploaded_files)
            prompt = hub.pull("rlm/rag-prompt")
            # create our Q&A chain
            chat_history = []
            pdf_qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={'k': 6}),
                return_source_documents=True,
                verbose=False
            )

            if prompt := st.chat_input(placeholder="Ask question about document"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                with st.chat_message("assistant"):
                    #st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                    with st.spinner("Searching for Answer..."):
                        result = pdf_qa.invoke(
                            {"question": prompt, "chat_history": chat_history})
                        response = result["answer"] #rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})["answer"]# rag_chain.invoke(prompt)#, callbacks=[st_cb])
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.write(response)
                        st.sidebar.write("Document Sources:")
                        st.sidebar.info(result['source_documents'])
                        chat_history.append((prompt, result["answer"]))
            
    else:
        st.warning('Please select agent type!', icon="⚠️")
if __name__ == '__main__':
    main()
