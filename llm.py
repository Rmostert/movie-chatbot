import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# Create the LLM

llm = ChatGoogleGenerativeAI(
	api_key= st.secrets['GOOGLE_API_KEY'],
	model = st.secrets["GEMINI_MODEL"],
	temperature=0.2
	)

# Create the Embedding model

embeddings = OpenAIEmbeddings(
	openai_api_key = st.secrets['OPENAI_API_KEY']
	)



