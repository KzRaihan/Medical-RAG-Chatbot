# 🏥 Medical Chatbot — RAG-Based Medical Question Answering

> Provides evidence-grounded general health information from verified medical reference material using Retrieval-Augmented Generation (RAG), semantic retrieval, and Large Language Models.

---

## 📌 Problem

Access to understandable and timely general health information can be difficult for people who have limited access to healthcare professionals, particularly in underserved settings.

Traditional search-based approaches require users to identify appropriate medical terminology, search through multiple sources, and determine which information is relevant and reliable.

Large Language Models (LLMs) provide a conversational interface, but directly asking an LLM medical questions introduces an important reliability problem: the model may generate information that is inaccurate, incomplete, outdated, or unsupported by a reliable medical source.

**Real-world question:**

> *Can we build a conversational medical information system that retrieves relevant information from controlled medical sources and generates concise, evidence-grounded responses while avoiding unsupported medical claims?*

This project addresses the problem by building a **Retrieval-Augmented Generation (RAG) Medical Chatbot** that retrieves relevant information from a controlled medical knowledge base and provides the retrieved context to an LLM before generating a response.

The system is designed for **general health education and information**, not medical diagnosis or treatment decisions.

---

## 🎯 Objective

**Medical-RAG-Chatbot** is a Generative AI application that uses **Retrieval-Augmented Generation (RAG)** to provide general medical information from a controlled knowledge source.

The system processes medical PDF documents using **LangChain** and **PyPDFLoader**, divides the content into smaller chunks using **RecursiveCharacterTextSplitter**, generates semantic embeddings using **Hugging Face Sentence Transformers**, and stores the embeddings in a **Pinecone vector database** for similarity-based retrieval.

Relevant medical context is retrieved for a user's question and passed to **GPT-4o** through a controlled prompt. The LLM then generates a concise response based primarily on the retrieved evidence.

The project is designed to evolve from **Jupyter Notebook experimentation** into a production-oriented medical information application with retrieval evaluation, answer-quality evaluation, safety controls, API integration, and deployment capabilities.

---

# 🧠 Solution Approaches

## 4.1 Approach 1 — Traditional Machine Learning / Deep Learning

A traditional ML/DL approach could attempt to train a medical question-answering or classification model specifically for the target medical domain.

However, this approach introduces several challenges:

➔ Requires a sufficiently large and high-quality medical question-answer dataset.

➔ Requires domain-specific labeled data.

➔ Requires significant computational resources for model training.

➔ Requires retraining when new medical knowledge is introduced.

➔ Difficult to directly connect generated answers to specific source documents.

➔ A trained model may still produce unsupported information.

Therefore, a pure ML/DL approach is **not selected for the initial version**.

---

## 4.2 Approach 2 — Direct LLM Question Answering

Another approach is to directly provide the user's medical question to an LLM.

```text
User Question
      ↓
     LLM
      ↓
Medical Response
```

Although this provides a simple conversational interface, the approach has an important limitation:

The response is primarily dependent on the LLM's internal knowledge and may not be directly grounded in the project's approved medical sources.

Therefore, direct LLM generation is not selected as the primary architecture.

---

## 4.3 Approach 3 — Generative AI + RAG (Selected)

The selected approach is a **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of relying exclusively on the LLM's internal knowledge, the system retrieves relevant information from the medical knowledge base and provides that information as context to the LLM.

### Core Idea

```text
Medical Reference
      ↓
Document Processing
      ↓
Knowledge Base
      ↓
Retrieve Relevant Medical Information
      ↓
LLM
      ↓
Grounded Medical Response
```

The RAG architecture provides a separation between:

```text
Knowledge Retrieval
        +
Response Generation
```

This allows retrieval quality and answer quality to be evaluated independently.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────────┐
                         │   MEDICAL REFERENCES     │
                         │                          │
                         │     Medical PDF/Book     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │     DOCUMENT LOADER      │
                         │                          │
                         │      PyPDFLoader         │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    TEXT PROCESSING       │
                         │                          │
                         │ RecursiveCharacter       │
                         │ TextSplitter             │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │     EMBEDDING MODEL      │
                         │                          │
                         │ all-MiniLM-L6-v2         │
                         │ 384-dimensional vectors  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │      VECTOR STORE        │
                         │                          │
                         │        Pinecone          │
                         └────────────┬─────────────┘
                                      │
                                      │
══════════════════════════════════════╪══════════════════════════════
                                      │
                                 USER QUERY
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │     QUERY EMBEDDING     │
                         │                          │
                         │     Embedding Model      │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       RETRIEVER          │
                         │                          │
                         │   Similarity Search      │
                         │         Top-K = 3        │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    RELEVANT CONTEXT      │
                         │                          │
                         │ Retrieved Medical Chunks │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │     PROMPT TEMPLATE      │
                         │                          │
                         │ Context + Question       │
                         │ + Safety Instructions    │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │           LLM            │
                         │                          │
                         │          GPT-OSS-20B     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    RESPONSE GENERATION   │
                         │                          │
                         │ Grounded Medical Answer  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   SAFETY / SCOPE CONTROL │
                         │                          │
                         │ Diagnosis / Treatment    │
                         │ Boundary Checking        │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │      FINAL RESPONSE      │
                         │                          │
                         │ General Health Education │
                         └──────────────────────────┘
```

---

# 🧩 Tech Stack

| Component               | Tool                               |
| ----------------------- | ---------------------------------- |
| Project Type            | Generative AI / RAG                |
| Programming Language    | Python                             |
| Orchestration Framework | LangChain                          |
| Document Loader         | PyPDFLoader                        |
| Text Splitter           | RecursiveCharacterTextSplitter     |
| Embedding Model         | Hugging Face Sentence Transformers |
| Embedding Model         | `all-MiniLM-L6-v2`                 |
| Embedding Dimension     | 384                                |
| Vector Store            | Pinecone                           |
| Similarity Metric       | Cosine Similarity                  |
| Retriever               | LangChain VectorStore Retriever    |
| LLM                     | GPT-4o                             |
| Prompt Engineering      | LangChain Prompt Templates         |
| Knowledge Source        | Medical PDF / Book                 |
| Experimentation         | Jupyter Notebook                   |
| API Layer               | FastAPI — Phase 2                  |
| Version Control         | Git & GitHub                       |

---

# 🔧 Development Phases

The project is divided into two major phases:

## Phase 1 — Experimentation (Jupyter Notebook)

### 1. Objective

The objective of Phase 1 is to experimentally develop and validate the core components of the **Medical-RAG-Chatbot** before moving to a production-oriented GenAI pipeline.

In this phase, the complete RAG workflow is implemented and tested inside a Jupyter Notebook.

Each component is evaluated independently to ensure that:

* medical documents can be loaded;
* text can be appropriately split;
* embeddings can be generated;
* the vector database can be created;
* relevant medical information can be retrieved;
* the LLM can generate responses using retrieved context;
* unsupported questions can be handled appropriately.

---

## 2. Phase 1 Workflow

```text
Medical PDF
      ↓
Document Loading
      ↓
Text Splitting
      ↓
Text Embedding
      ↓
Pinecone Vector Store
      ↓
Retriever
      ↓
Relevant Medical Context
      ↓
Prompt Construction
      ↓
LLM
      ↓
Grounded Medical Response
      ↓
Safety / Scope Control
      ↓
Evaluation
```

---

## 3. Load Medical PDF Document

* The project initially uses a medical reference PDF as the primary knowledge source.

* The PDF is loaded using `PyPDFLoader`.

* The loaded PDF is converted into LangChain `Document` objects containing:

  * Page content

  * Source information

  * Page metadata

### Current Experiment

The experimental medical reference contains approximately:

```text
637 pages
```

---

## 4. Document Metadata Processing

The loaded documents contain page-level metadata.

The experiment extracts relevant information such as:

```text
page_content
source
```

This allows the system to maintain a relationship between retrieved information and its original document source.

---

## 5. Text Splitting

Large medical documents cannot efficiently be processed as a single text block.

Therefore, the document is divided into smaller overlapping chunks.

### Current Configuration

```text
Chunk Size     = 500
Chunk Overlap  = 20
```

The current experiment produces approximately:

```text
5,859 chunks
```

### Why Text Splitting?

Smaller chunks allow the retrieval system to identify more specific pieces of medical information instead of retrieving an entire page or document.

The overlap helps preserve contextual continuity between neighboring chunks.

---

## 6. Initialize Embedding Model

The next step is to convert each medical text chunk into a numerical vector representation.

The experiment uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model generates:

```text
384-dimensional embeddings
```

Semantic information contained in the medical text is represented in vector space, allowing similar questions and medical passages to be compared using vector similarity.

---

## 7. Generate Document Embeddings

The document chunks are converted into embeddings and stored in the vector database.

### Architecture

```text
Medical PDF
     ↓
Medical Chunks
     ↓
Embeddings
     ↓
Pinecone
```

The document knowledge base has now been transformed into a searchable semantic vector index.

---

## 8. Initialize Pinecone Vector Store

Pinecone is used as the vector database for storing and retrieving medical document embeddings.

### Current Experiment Configuration

```text
Index Name       = medical-bot
Dimension        = 384
Metric           = cosine
Cloud            = AWS
Region           = us-east-1
```

The vector database provides persistent storage for the medical knowledge representation.

---

## 9. Upsert Medical Embeddings

The generated document embeddings are inserted into the Pinecone index.

The conceptual workflow is:

```text
Medical Chunk
     ↓
Embedding
     ↓
Vector + Metadata
     ↓
Pinecone Index
```

The associated metadata allows retrieved vectors to be mapped back to their source document content.

---

## 10. Load Existing Vector Store

The project also tests loading the previously created Pinecone index.

This avoids processing the complete medical PDF and regenerating embeddings every time the application starts.

### Benefit

```text
First Run
PDF → Chunk → Embed → Pinecone

Later Runs
Existing Pinecone Index
        ↓
     Retriever
```

This makes the system more suitable for a production pipeline.

---

## 11. Create Retriever

The Pinecone vector store is converted into a retriever.

The retriever is responsible for finding medical document chunks relevant to a user's question.

### Retrieval Strategy

```text
User Query
     ↓
Query Embedding
     ↓
Pinecone Similarity Search
     ↓
Top-K Relevant Chunks
```

### Current Configuration

```text
Search Type = similarity
k = 3
```

---

## 12. Test Medical Document Retrieval

The retriever is tested using representative medical questions.

For example:

```text
What is Acne?
```

The retriever should return medical chunks containing information relevant to Acne.

This step is important because the quality of the final RAG response depends heavily on whether the correct evidence is retrieved.

If irrelevant chunks are returned, the retrieval configuration, chunking strategy, embedding model, or query-processing strategy may require improvement.

---

## 13. Initialize LLM

The LLM is initialized as the response-generation component.

### Current Experiment

```text
LLM = GPT-4o
```

The LLM does not receive the complete medical book for every question.

Instead, it receives:

```text
User Question
      +
Retrieved Medical Context
```

This is the central mechanism of Retrieval-Augmented Generation.

---

## 14. Create Medical Response Prompt

A prompt template is designed to instruct the LLM to answer using the retrieved medical context.

### Requirements

* Use the provided medical context as the primary knowledge source.
* Answer the user's question directly.
* Do not introduce unsupported medical claims.
* If the required information is not available, indicate that the information is unknown or insufficient.
* Keep the response concise and understandable.
* Do not present the system as a replacement for a healthcare professional.
* Do not provide definitive diagnosis or individualized treatment decisions.

---

## 15. Create RAG Chain

The retrieved context and user question are combined into a RAG pipeline.

Conceptually:

```text
User Question
      ↓
Retriever
      ↓
Relevant Medical Context
      ↓
Prompt
      ↓
GPT-4o
      ↓
Medical Response
```

The current experiment uses LangChain's retrieval and document-combination components to construct this workflow.

---

## 16. Generate Medical Response

The LLM receives the retrieved medical context together with the user's question.

For example:

```text
Question:
What is Acne?
```

The system:

```text
Question
   ↓
Retrieve Relevant Medical Chunks
   ↓
Insert Chunks into Prompt
   ↓
GPT-4o
   ↓
Generate Context-Grounded Answer
```

---

## 17. Test RAG Responses

The complete RAG chain is tested using medical questions.

Example categories include:

```text
General Medical Knowledge
        ↓
Disease Information
        ↓
Symptoms
        ↓
Medical Terminology
        ↓
Treatment Information
```

The generated answer should be evaluated against the retrieved evidence rather than simply checking whether the response sounds medically appropriate.

---

## 18. Test Out-of-Knowledge Questions

The system should also be tested with questions for which sufficient information does not exist in the knowledge base.

The expected behavior is:

```text
User Question
      ↓
Insufficient Relevant Evidence
      ↓
Do Not Invent Information
      ↓
Appropriate Abstention
```

This experiment is important for evaluating whether the RAG system can avoid unsupported generation.

---

## 19. Medical Safety and Scope Control

Because this is a medical information system, explicit boundaries are required.

The chatbot should distinguish between:

```text
General Health Education
          ↓
      In Scope

Definitive Diagnosis
          ↓
      Out of Scope

Personalized Treatment Decision
          ↓
      Out of Scope

Medication Prescription / Dosage
          ↓
      Out of Scope
```

The system should provide appropriate scope-limited responses for questions that require clinical assessment.

---

## 20. Evaluate Retrieval Quality

Retrieval should be evaluated independently from LLM generation.

The evaluation dataset should contain:

```text
Question
Expected Relevant Information
Expected Source / Passage
```

Possible retrieval metrics include:

| Metric              | Purpose                                                |
| ------------------- | ------------------------------------------------------ |
| Recall@k            | Whether relevant evidence appears in the top-k results |
| Precision@k         | Proportion of retrieved results that are relevant      |
| MRR                 | Rank of the first relevant result                      |
| Retrieval Relevance | Human assessment of retrieved context                  |

Multiple values of `k` should be tested rather than assuming that `k=3` is optimal.

---

## 21. Evaluate Generated Answers

Generated responses should be evaluated against the retrieved evidence.

Important evaluation dimensions include:

| Metric                 | Purpose                                           |
| ---------------------- | ------------------------------------------------- |
| Faithfulness           | Whether claims are supported by retrieved context |
| Correctness            | Whether the answer is medically correct           |
| Relevance              | Whether the answer addresses the question         |
| Completeness           | Whether important information is covered          |
| Unsupported Claim Rate | Percentage of claims without supporting evidence  |
| Response Latency       | Time required to generate the answer              |

---

## 22. Evaluate Safety Behavior

A dedicated safety test set should contain:

* normal medical education questions;
* ambiguous symptom questions;
* questions outside the knowledge base;
* diagnosis requests;
* personalized treatment requests;
* medication-related questions;
* questions requiring professional medical evaluation.

Expected behavior:

```text
Supported Medical Question
          ↓
   Evidence-Grounded Answer


Insufficient Evidence
          ↓
   Appropriate Abstention


Out-of-Scope Clinical Request
          ↓
   Safe Scope-Limited Response
```

---

# 23. Phase 1 Experimentation Results

At the end of Phase 1, the following components should be successfully validated:

| Component                   | Status |
| --------------------------- | ------ |
| Medical PDF Loading         | ☐      |
| Metadata Processing         | ☐      |
| Text Splitting              | ☐      |
| Chunk Analysis              | ☐      |
| Embedding Generation        | ☐      |
| Pinecone Vector Store       | ☐      |
| Vector Indexing             | ☐      |
| Retriever                   | ☐      |
| Retrieval Testing           | ☐      |
| LLM Integration             | ☐      |
| RAG Chain                   | ☐      |
| Medical Response Generation | ☐      |
| Out-of-Knowledge Testing    | ☐      |
| Safety Testing              | ☐      |
| Retrieval Evaluation        | ☐      |
| Answer Evaluation           | ☐      |

---

# 24. Experimentation Observations

During experimentation, important observations should be recorded for each component.

## Document Loading

```text
Observation:

The medical PDF was successfully loaded and converted into
LangChain Document objects containing page content and metadata.
```

## Text Splitting

```text
Observation:

The medical reference was divided into smaller overlapping chunks
to improve semantic retrieval and LLM context management.

Current result:
Approximately 5,859 chunks were generated.
```

## Embedding

```text
Observation:

Medical document chunks were converted into 384-dimensional
semantic vector representations using all-MiniLM-L6-v2.
```

## Vector Store

```text
Observation:

Pinecone successfully stored the medical document embeddings
and enabled semantic similarity-based retrieval.
```

## Retrieval

```text
Observation:

The retriever successfully returned relevant medical document
chunks for representative medical questions.

Current retrieval configuration:
Similarity Search, k = 3.
```

## LLM Generation

```text
Observation:

GPT-4o generated responses using the retrieved medical context
provided by the RAG pipeline.
```

## RAG Pipeline

```text
Observation:

The complete retrieval → context → prompt → LLM workflow
successfully generated medical responses from retrieved evidence.
```

## Safety

```text
Observation:

The system requires explicit testing for unsupported questions,
diagnosis requests, personalized treatment requests, and other
clinical use cases outside the intended scope.
```

---

# 25. Phase 1 → Phase 2 Transition

Once the complete RAG pipeline has been validated in Jupyter Notebook, the implementation will move to:

**Phase 2 — Production GenAI Pipeline**

### Phase 1

```text
Jupyter Notebook
      ↓
Experiment
      ↓
Validate Components
      ↓
Evaluate Retrieval
      ↓
Evaluate Generation
      ↓
Establish Safety Rules
      ↓
Find Best Configuration
```

The primary objective of Phase 1 is therefore not simply to demonstrate a working chatbot.

It is to:

> **Experimentally determine a reliable retrieval and generation configuration that can later be transformed into a maintainable medical information application.**

---

# Phase 2 — Production Pipeline (FastAPI)

After validating the individual components in Jupyter Notebook, the system will be converted into a production-oriented application.

## Production GenAI Pipeline

```text
                         ┌──────────────────────┐
                         │        USER          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI API     │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
             Upload Medical PDF                 User Query
                    │                                │
                    ▼                                │
             Document Pipeline                       │
                    │                                │
                    ▼                                │
            Chunking + Embedding                     │
                    │                                │
                    ▼                                │
                 Pinecone ◄──────────────────────────┘
                    │
                    ▼
              Retrieval Layer
                    │
                    ▼
             Relevant Context
                    │
                    ▼
             Prompt Pipeline
                    │
                    ▼
                 LLM Layer
                    │
                    ▼
          Medical Response Generation
                    │
                    ▼
           Safety / Scope Validation
                    │
                    ▼
             Final Response
```

---

# 🛡️ Production Safety Layer

The production version should introduce additional controls around the RAG pipeline.

```text
                     User Query
                          │
                          ▼
                 Query Classification
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
        Educational    Clinical     Unsafe /
          Query         Request      Unsupported
             │            │            │
             ▼            ▼            ▼
         RAG Answer   Scope-Limited   Safe
                       Response      Response
             │            │            │
             └────────────┴────────────┘
                          │
                          ▼
                    Final Output
```

The purpose of this layer is not to make clinical decisions, but to prevent the application from presenting itself as a diagnostic or treatment system.

---

# 📊 Future Evaluation Pipeline

The production-oriented version should support continuous evaluation.

```text
User Question
      ↓
Retrieval
      ↓
Retrieved Context
      ↓
LLM Response
      ↓
Evaluation
      │
      ├── Retrieval Relevance
      ├── Faithfulness
      ├── Correctness
      ├── Completeness
      ├── Unsupported Claims
      ├── Safety Behavior
      └── Response Latency
```

This allows the project to move beyond a simple chatbot demonstration toward an experimentally evaluated RAG system.

---

# 📦 Expected Project Deliverables

The completed project is expected to produce:

1. **Curated medical knowledge base**
2. **Medical document-processing pipeline**
3. **Text chunking pipeline**
4. **Embedding generation pipeline**
5. **Pinecone vector database**
6. **Semantic retrieval component**
7. **RAG response-generation pipeline**
8. **Medical prompt templates**
9. **Safety and scope-control rules**
10. **Retrieval evaluation dataset**
11. **Answer evaluation dataset**
12. **Retrieval evaluation results**
13. **Answer-quality evaluation results**
14. **Safety evaluation results**
15. **Error analysis**
16. **FastAPI backend**
17. **Working Medical-RAG-Chatbot application**

---

# 📋 Project Scope

### In Scope

```text
✓ General medical information
✓ Medical terminology explanation
✓ Disease education
✓ General symptom information
✓ Evidence retrieval
✓ RAG-based question answering
✓ Source-grounded response generation
✓ Retrieval evaluation
✓ Answer evaluation
✓ Safety/scope evaluation
```

### Out of Scope

```text
✗ Definitive medical diagnosis
✗ Personalized treatment decisions
✗ Medication prescription
✗ Medication dosage decisions
✗ Medical imaging diagnosis
✗ Replacement for healthcare professionals
✗ Autonomous clinical decision-making
✗ Clinical validation
```

---

# 🔬 Expected Research Contribution

The primary contribution of the project is **not the development of a new Large Language Model**.

Instead, the project investigates the design and evaluation of a **source-grounded medical question-answering system**.

The research focus is:

```text
Medical Knowledge
       +
Semantic Retrieval
       +
Evidence Grounding
       +
LLM Generation
       +
Safety Constraints
       ↓
Reliable Medical Information System
```

The project will investigate how retrieval quality, chunking strategy, embedding models, retrieval configuration, prompt design, and safety controls affect the quality and reliability of generated medical responses.

---

# 📝 Project Summary

**Medical-RAG-Chatbot** is a Generative AI application that uses **Retrieval-Augmented Generation (RAG)** to provide evidence-grounded general medical information from controlled medical references.

The system processes medical PDF documents using **LangChain** and **PyPDFLoader**, divides the content into smaller chunks using **RecursiveCharacterTextSplitter**, generates semantic embeddings using **Hugging Face Sentence Transformers**, and stores the embeddings in a **Pinecone vector database** for similarity-based retrieval.

Relevant medical context is retrieved for a user's question and passed to **GPT-4o** through a controlled prompt. The generated response is intended to remain grounded in the retrieved medical evidence and within explicitly defined safety boundaries.

The project is designed to evolve from **Jupyter Notebook experimentation** into a production-oriented GenAI application with **retrieval evaluation, answer-quality evaluation, safety validation, FastAPI integration, and deployment capabilities**.
