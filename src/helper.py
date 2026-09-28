# =================================================================
# - Define the All Helper relative function such as
# document loader, text splitter, embedding , vector store
# =================================================================


# import Libraries
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings




# Extract Data
def load_pdf_file(data):
    """
    Function : load_pdf_file
        - purpose: Split the all pdf data into single document
        
        - input: Directory of pdf file path
        
        - output: All Documents into single docs file
    """
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )
    documents = loader.load()

    return documents

#### Organize the All Document__ only count the page_content and source in our Document
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Function : filter_to_minimal_docs
        - purpose: Filter the document (only extract the page_content and source)
        
        - input: list of Document that include some metadata
        
        - output: Filtered of list of Document that have only page_content and source
    """
    minimal_docs: List[Document] = []

    # where, docs : list of old document that have some extra information
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source":src}

            )
        )

    return minimal_docs


# split the data into Text chunks
def text_split(minimal_docs):
    """
    Function : text_split
        - purpose: split the minimal docs document into smaller chunk 
        
        - input: list of Document
        
        - return: list of chunk document
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 25)

    text_chunks = text_splitter.split_documents(minimal_docs)

    return text_chunks



### 4. Download the Embedding model form the Hugging Face
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings

