from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch, helpers
from langchain_community.vectorstores import ElasticsearchStore
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback
import os


# Sidebar contents
with st.sidebar:
    st.title('💬PDF Summarizer and Q/A App')
    st.markdown('''
    ## About this application
    You can built your own customized LLM-powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(2)
    st.write(' Why drown in papers when your chat buddy can give you the highlights and summary? Happy Reading. ')
    add_vertical_space(2)    

def main():
    load_dotenv()
    OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
    ES_USER = os.getenv("ES_USER")
    ES_PASSWORD = os.getenv("ES_PASSWORD")
    ES_ENDPOINT = os.getenv("ES_ENDPOINT")
    elastic_index_name='pdf_docs'

    #Main Content
    st.header("Ask About Your PDF 🤷‍♀️💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF File and Ask Questions", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # Make a connection to Elasticsearch
      
      url = f"https://{ES_USER}:{ES_PASSWORD}@{ES_ENDPOINT}:9200"
 
      connection = Elasticsearch(
        hosts=[url], 
        ca_certs = "./http_ca.crt", 
        verify_certs = True
      )
      print(connection.info())
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      if not connection.indices.exists(index=elastic_index_name):
        print("The index does not exist, going to generate embeddings")   
        docsearch = ElasticsearchStore.from_texts( 
                chunks,
                embedding = embeddings, 
                es_url = url, 
                es_connection = connection,
                index_name = elastic_index_name, 
                es_user = ES_USER,
                es_password = ES_PASSWORD
        )
      else: 
        print("The index already existed")
        
        docsearch = ElasticsearchStore(
            es_connection=connection,
            embedding=embeddings,
            es_url = url, 
            index_name = elastic_index_name, 
            es_user = ES_USER,
            es_password = ES_PASSWORD    
        )

            # show user input
      with st.chat_message("user"):
        st.write("Hello World 👋")
      user_question = st.text_input("Please ask a question about your PDF here:")
      if user_question:
        docs = docsearch.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
        
if __name__ == '__main__':
    main()