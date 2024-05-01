#PIP COMMANDS REQUIRED TO USE
# !pip install langchain openai weaviate-client
# !pip install tiktoken

import glob
import os
#PUT APT KEY HERE
os.environ["OPENAI_API_KEY"] = ""

from langchain.document_loaders import TextLoader
from google.colab import drive

drive.mount('/content/drive')

document_files = glob.glob('/content/drive/MyDrive/DatasetsClean/CSVs/*.csv')
documents = []
for file_path in document_files:
    loader = TextLoader(file_path)
    documents.extend(loader.load())

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#all_chunks = [chunk for document in documents for chunk in text_splitter.split_document(document)]
chunks = text_splitter.split_documents(documents)

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions



client = weaviate.Client(
  embedded_options = EmbeddedOptions()
)

vectorstore = Weaviate.from_documents(
    client = client,
    documents = chunks,
    embedding = OpenAIEmbeddings(),
    by_text = False
)

retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)

from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# query = "List all of the unique creation times found within the provided exif metadata provided in the context."
# rag_chain.invoke(query)

query = input("Please enter your question: ")

# Invoke the chain with the user query and print the response
response = rag_chain.invoke(query)
print(response)
