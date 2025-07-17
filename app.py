Q: What are your business hours?
A: Our support team is available Monday to Friday, 9am to 5pm.

Q: How can I reset my password?
A: You can reset your password by clicking "Forgot password" on the login page.

Q: Do you offer refunds?
A: Yes, we offer full refunds within 14 days of purchase if you're not satisfied.

Q: How do I contact customer support?
A: You can email us at support@example.com or call us at +123456789.
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
