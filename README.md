app.py
import os
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Load FAQ data
loader = TextLoader("faq.txt")
documents = loader.load()

# Split and embed text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Build Retrieval QA chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=db.as_retriever())

# Example query loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa.run(query)
    print(f"\nAI Answer: {result}")
