from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import openai

from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from elasticsearch import Elasticsearch, helpers
from langchain_elasticsearch import ElasticsearchStore
from langchain.callbacks import get_openai_callback

from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import AzureOpenAI
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI

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
      
      model_name = os.getenv('MODEL_NAME')
      azure_embedding_endpoint = os.getenv('AZURE_EMBEDDING_ENDPOINT')
      azure_embedding_api_key = os.getenv('AZURE_EMBEDDING_API_KEY')
      azure_embedding_api_version = os.getenv("AZURE_EMBEDDING_API_VERSION")
      
      # create embeddings
      embeddings = AzureOpenAIEmbeddings(
        model=model_name,
        azure_endpoint=azure_embedding_endpoint, 
        api_key= azure_embedding_api_key,
        openai_api_version=azure_embedding_api_version
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
      
      azure_api_key = os.getenv('AZURE_API_KEY')
      azure_endpoint = os.getenv('AZURE_EDNPOINT')
      azure_api_version = os.getenv('AZURE_API_VERSION')
      azure_deployment_id = os.getenv('AZURE_DEPLOYMENT_ID')
      
      llm = AzureChatOpenAI(
          api_key=azure_api_key,  
          api_version=azure_api_version,
          azure_endpoint=azure_endpoint,
          azure_deployment=azure_deployment_id,
      )           
      
      # show user input
      with st.chat_message("user"):
        st.write("Hello World 👋")
      user_question = st.text_input("Please ask a question about your PDF here:", "what is the summary of the pdf?")
      if user_question:
        docs = docsearch.similarity_search(user_question)
              
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
            
        st.write(response)
        
if __name__ == '__main__':
    main()