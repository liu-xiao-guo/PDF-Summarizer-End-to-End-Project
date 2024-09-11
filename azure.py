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

import openai
from langchain_openai import AzureOpenAIEmbeddings
# from openai import AzureOpenAI
from langchain.llms import AzureOpenAI
# from langchain import AzureOpenAI
# from langchain.chains import load_qa_chain
# from langchain.chains.qa import load_qa_chain
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
    
    # OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
    # BASE_URL=os.getenv("BASE_URL")
    # API_TYPE=os.getenv("API_TYPE")
    # API_VERSION=os.getenv("API_VERSION")
    
    openai.api_key="6ef02367da1143b8bea6a4c1762d3e56"
    openai.api_base="https://embeddings-testing1.openai.azure.com/"
    openai.api_type="azure"
    openai.api_version="2023-05-15"
    # print("api key: ", OPENAI_API_KEY)
    
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
      embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_endpoint=openai.api_base, 
        api_key= openai.api_key,
        openai_api_version=openai.api_version
      )
      
      # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
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
        
      openai.api_type = "azure"
      openai.api_key = "6ef02367da1143b8bea6a4c1762d3e56"  # Replace with your Azure API key
      openai.api_base = "https://embeddings-testing1.openai.azure.com/"  # Replace with your Azure endpoint
      openai.api_version = "2023-03-15-preview"  # Example API version
      deployment_id = "gpt-4o"
      
      llm = AzureOpenAI(
            openai_api_key=openai.api_key, 
            deployment_name=deployment_id, 
            model_name="gpt-4o",
            api_key=openai.api_key,
            api_base=openai.api_base,
            api_version=openai.api_version,
            )    
        
      # llm = AzureOpenAI(
      #     api_key=openai.api_key,  
      #     api_version=openai.api_version,
      #     azure_endpoint=openai.api_base,
      #     azure_deployment=deployment_id
      # )           
      # show user input
      with st.chat_message("user"):
        st.write("Hello World 👋")
      user_question = st.text_input("Please ask a question about your PDF here:")
      if user_question:
        docs = docsearch.similarity_search(user_question)
              
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
            
        st.write(response)
        
if __name__ == '__main__':
    main()