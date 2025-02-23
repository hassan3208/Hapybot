from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
import os
import faiss 


temp=0.2


 # create embedding
    
embeddings = OllamaEmbeddings(
model="deepseek-r1:1.5b",
)    

# create a vectorstore

d = len(embeddings.embed_query("test query"))  # Dimension of the embedding (check the model you're using)
faiss_index = faiss.IndexFlatIP(d)

vector_store = FAISS(
embedding_function=embeddings,
index=faiss_index,
docstore=InMemoryDocstore(),
index_to_docstore_id={},
)







                         ###################################################
                         #                  GET CHATBOT RESPONSE           #
                         ###################################################
                         

def get_bot_response(QUERY, results):
    llm = ChatOllama(
        model="deepseek-r1:1.5b",
        temperature=temp,
    )

    messages = [
        ("system", "You are an AI assistant that answers user queries based on provided documents or webpage links."),
        ("system", "If the user asks an irrelevant query, respond with 'No relevant information found'—no other words."),
        ("system", "Keep your answers concise."),
        ("system", "Add one dark sexual joke at the end."),
    ]
    
    for res in results:
        messages.append(("system", res))  # Dynamically add retrieved chunks

    messages.append(("human", QUERY))
    
    ai_msg = llm.invoke(messages)
    return ai_msg.content











                         
                         ###################################################
                         #                  GET SIMILAR CHUNKS             #
                         ###################################################
                         

def GET_SIMILAR_CHUNK(QUERY):
    query_embedding = embeddings.embed_query(QUERY)  # Convert query to embedding
    results = vector_store.similarity_search_by_vector(query_embedding, k=3)
    
    return [res.page_content for res in results]

 

    











                         
                         ###################################################
                         #                  SETUP A VECTORSTORE            #
                         ###################################################
                         


def PASS_DATA_INTO_VECTORSTORE(CHUNK_LIST):
    texts = [text for text in CHUNK_LIST]
    metadatas = [{"source": "pdf"} for _ in CHUNK_LIST]

    # Bulk embedding
    embeddings_list = embeddings.embed_documents(texts)

    # Add texts and precomputed embeddings
    vector_store.add_texts(texts=texts, metadatas=metadatas, embeddings=embeddings_list)













                         
                         ###################################################
                         #                  GET SPLITTED TEXT              #
                         ###################################################
                         
                         

def SPLIT_TEXT(TEXT):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=20)
    return text_splitter.split_text(TEXT)





                         
                         ###################################################
                         #                  GET URL TEXT                   #
                         ###################################################
                         
                         
def GET_URL_TEXT(URL):
    loader = WebBaseLoader(URL)

    docs=loader.load()
    return docs[0].page_content







                         
                         ###################################################
                         #                  GET PDF TEXT                   #
                         ###################################################
                         


def GET_PDF_TEXT(PDF):
    loader = PyPDFLoader(
        PDF,
        mode="page",
        images_inner_format="markdown-img",
        images_parser=RapidOCRBlobParser(),
    )
    docs = loader.load()
    
    pdf_text=''
    for i,doc in enumerate(docs):
        pdf_text+=doc.page_content
        
    return pdf_text






                         
                         ###################################################
                         #                  GET WORD TEXT                  #
                         ###################################################
                        

def GET_WORD_TEXT(WORD):
    loader = Docx2txtLoader(WORD)
    docs = loader.load()
    
    word_text=''
    for i,doc in enumerate(docs):
        word_text+=doc.page_content
        
    return word_text




                         
                         ###################################################
                         #                  GET CSV TEXT                   #
                         ###################################################
                         



def GET_CSV_TEXT(CSV_FILE):
    loader = CSVLoader(
    file_path=CSV_FILE,
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
    },
    )

    data = loader.load()

    csv_text = "\n".join([doc.page_content for doc in data])

    return csv_text
