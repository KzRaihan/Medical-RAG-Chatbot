## ========================================================================
#   Create a Pinecone Vector Database
#   - create an index and store the embeddings into pinecone db
## ========================================================================
import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, filter_to_minimal_docs ,text_split, download_hugging_face_embeddings

from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Read the Pinecone API key from .env file
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

# save pinecone Api Key into our environment 
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# 1. Load the Pdf
extracted_data = load_pdf_file(data="data/")

# 2. Filter to minimal docs
minimal_docs = filter_to_minimal_docs(extracted_data)

# 3. Chunking
text_chunks = text_split(minimal_docs)

# 4. Download the embedding model
embeddings = download_hugging_face_embeddings()

# 5. Create a pinecone vector database
#  Load API key from environment
api_key = os.getenv("PINECONE_API_KEY")

# Initialize modern SDK client    
pc = Pinecone(api_key=api_key)

# Step 1: Create an index 
index_name = "medical-bot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = "cosine",
        spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
    )

# Connect to Pinecone Index
index = pc.Index(index_name)

#### Step 2: Store Data into Pinecone Vector DataBase
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name = index_name,
    embedding=embeddings
)

