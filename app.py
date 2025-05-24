
import streamlit as st
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO
from PyPDF2.errors import PdfReadError
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
#import google.generativeai as genai

from langchain_community.vectorstores import FAISS
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import langchain_openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv




load_dotenv()
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    texts = []
    for pdf in pdf_docs:
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            texts.append(text)
        except PdfReadError:
            print("Skipping file: Invalid or corrupt PDF.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return str(texts)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks,embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    Prompt_template = """ 
    Answer the questions as detailed as possible from the the provided context,make sure to answer only from the provided context. If answer is not provided in the
    context, clearly mention that the answer is not available in the context. 
    Context:\n{context}?\n
    Questions:\n{questions}\n
    Answer:
    """
    model =ChatOpenAI(model="gpt-4",temperature=0.3)
    prompt = PromptTemplate(template=Prompt_template,input_variables=["context","questions"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain
def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs =new_db.similarity_search("user_question")
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs,"questions":user_question},
        return_only_outputs=True)
    
    print(response)
    st.write("Reply: ",response["output_text"])

def main():
    st.set_page_config("Chat with multiple pdf")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a question from the PDF Files")

    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your pdf files",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing...."):
                raw_text = get_pdf_text(pdf_docs)
                print("Type of raw_text:", type(raw_text))
                print("Value of raw_text:", raw_text)
                if not isinstance(raw_text, str):
                    raise ValueError("raw_text must be a string")
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ =="__main__":
    main()




