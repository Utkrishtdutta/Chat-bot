'''
An simple steamlit application through which you can question your PDF and find answer related to it.
Tech Stack: Streamlit
            Hugging Face
            Langchain
            Gemma(open source LLMs)
            Chroma (open source Vector DB)

'''
import streamlit as st
from langchain.vectorstores  import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from llms import generate_gemma,generate_openai,read_file


st.title('Question the PDF')


file = st.file_uploader(label='Upload Your documents',type=['pdf','docx','txt'])
if file:
    file_content = read_file(file)
    text_file =  open('file.txt','w')
    text_file.write(file_content)
    text_file.close()
    loader = TextLoader('file.txt',encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding_function)


    query = st.text_input('write your question here')
    ask = st.button('ASK LLMs')
    if ask:
        docs = db.similarity_search(query)
        text_reference = docs[0].page_content
        prompt = f'{text_reference} \n Questions:{query}'
        response = generate_openai(prompt)
        st.text_area('The response:- ', response)
