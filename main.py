import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import methods as func
import random
import time
import os
os.environ["USER_AGENT"] = "MyAssistantBot/1.0"



vector_store=func.vector_store

#  # create embedding
    
# embeddings = OllamaEmbeddings(
# model="deepseek-r1:1.5b",
# )    

# # create a vectorstore

# vector_store = FAISS(
# embedding_function=embeddings,
# index=[],
# docstore=InMemoryDocstore(),
# index_to_docstore_id={},
# )



st.set_page_config(
    page_title="HAPPY BOT",  # Sets the tab name
    page_icon="🤖",  # Sets the tab icon (can use emoji or URL of an image)
    # layout="wide",  # Optional: makes the layout wider
)

st.title("HAPPY BOT 🤖")




st.write("This is a simple ChatBot that will answer your quries based on your uploaded document or web links.")

st.caption("Note that this app is connected with Chinese DEEPSEEK AI if you want to run it please visit my GITHUB and read PRE-REQS")



# Sidebar for uploading documents and entering URLs
with st.sidebar:
    st.header("📂 Upload Document or 🌍 Enter Website URL\n\n")
    
    selected_value = st.slider("adjust creativity factor", min_value=0, max_value=100, value=20)
    st.write(f"Your creativity : {selected_value}%")
    func.temp=selected_value/100
    
    # Options to choose either file upload or URL input
    option = st.radio("Select input type:", ["Upload Document", "Enter Website URL"])

    uploaded_file = None
    website_urls = None

    if option == "Upload Document":
        uploaded_files = st.file_uploader("Choose a file", type=["pdf", "csv", "docx"],accept_multiple_files=True)
    
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type
    
                if file_type == "application/pdf":
                    # Save PDF temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
    
                    # Extract text
                    
                    extracted_text = func.GET_PDF_TEXT(temp_path)
                    splitted_text=func.SPLIT_TEXT(extracted_text)
                    st.write(splitted_text)
                    func.PASS_DATA_INTO_VECTORSTORE(CHUNK_LIST=splitted_text)
                    
                    
                    
                    os.remove(temp_path)  # Delete file after reading
                    
                    st.write(f"📜 Extracted text from {uploaded_file.name}:")
                    # st.text_area(f"📜 Extracted text from {uploaded_file.name}", extracted_text, height=200, key=uploaded_file.name)

    
                elif file_type == "text/csv":
                    # Save PDF temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
    
                    # Extract text
                    extracted_text = func.GET_CSV_TEXT(temp_path)
                    splitted_text=func.SPLIT_TEXT(extracted_text)
                    st.write(splitted_text)
                    func.PASS_DATA_INTO_VECTORSTORE(CHUNK_LIST=splitted_text)
                    
                    
                    os.remove(temp_path)  # Delete file after reading
                    
                    st.write(f"📜 Extracted text from {uploaded_file.name}:")
                    # st.text_area(f"📜 Extracted text from {uploaded_file.name}", extracted_text, height=200, key=uploaded_file.name)
    
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    # Save PDF temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
    
                    # Extract text
                    extracted_text = func.GET_WORD_TEXT(temp_path)
                    splitted_text=func.SPLIT_TEXT(extracted_text)
                    st.write(splitted_text)
                    func.PASS_DATA_INTO_VECTORSTORE(CHUNK_LIST=splitted_text)
                    
                    
                    os.remove(temp_path)  # Delete file after reading
                    
                    st.write(f"📜 Extracted text from {uploaded_file.name}:")
                    # st.text_area(f"📜 Extracted text from {uploaded_file.name}", extracted_text, height=200, key=uploaded_file.name)
    
                else:
                    st.warning(f"❌ Unsupported file type: {file_type}")

    
    elif option == "Enter Website URL":
        website_urls = st.text_area("Enter web links (one per line)")
        if website_urls:
            extracted_text=func.GET_URL_TEXT(website_urls)
            splitted_text=func.SPLIT_TEXT(extracted_text)
            st.write(splitted_text)
            func.PASS_DATA_INTO_VECTORSTORE(splitted_text)
    
    if st.button("Submit"):
        if option == "Upload Document" and uploaded_file:
            st.session_state["submitted_file"] = uploaded_file
            st.success("Document uploaded successfully!")
        
        elif option == "Enter Website URL" and website_urls.strip():
            urls_list = website_urls.split("\n")
            st.session_state["submitted_urls"] = [url.strip() for url in urls_list if url.strip()]
            st.success(f"Submitted {len(st.session_state['submitted_urls'])} URL(s)!")
        
        else:
            st.warning("Please upload a document or enter at least one URL.")

    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. Let's start again! 👇"}]
        st.rerun()






# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    
    similar_vector=func.GET_SIMILAR_CHUNK(f'{prompt}')
    bot_reply=func.get_bot_response(prompt,similar_vector)
    
    
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = bot_reply
        
        
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
