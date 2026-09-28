from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


# 1. Initialize the Flask
app = Flask(__name__)

# 2. Load the Credential
load_dotenv()
# Read the Pinecone API key from .env file
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

# save Groq and pinecone Api Key into our environment 
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# 3. call the embedding model
embeddings = download_hugging_face_embeddings()

# 4. Load the Existing index
index_name = "medical-bot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# 5. Create an vector retriever
retriever = docsearch.as_retriever(
    search_type = "similarity", # By Default
    search_kwargs = {
        "k": 3,
    }
)

# 6. Initialize the model
chatModel = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1
)

# 7. Define the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# 8. Create Chain

# ------------------------------------------------------------
# Helper to combine retrieved documents into a single context string
# ------------------------------------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ------------------------------------------------------------
# Equivalent to: create_stuff_documents_chain(chatModel, prompt)
#
# Takes {"context": [Document, ...], "input": "..."} and returns
# the LLM's generated answer as a plain string.
# ------------------------------------------------------------

question_answer_chain = (
    {
        "context": lambda x: format_docs(x["context"]),
        "input": lambda x: x["input"],
    }
    | prompt
    | chatModel
    | StrOutputParser()
)

# ------------------------------------------------------------
# Equivalent to: create_retrieval_chain(retriever, question_answer_chain)
#
# Step 1: retrieve documents for the input query, add them under "context"
# Step 2: pass the full dict (input + context) into question_answer_chain,
#         add the result under "answer"
# ------------------------------------------------------------

rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: retriever.invoke(x["input"]))
    )
    | RunnablePassthrough.assign(
        answer=question_answer_chain
    )
)




# 9. Create a default route
@app.route("/")
def index():
    return render_template('chat.html')

# 10. Create a route that invoke the chain
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response["answer"])  # only get the answer (not others meta data)
    return str(response["answer"])



# 6. Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True )