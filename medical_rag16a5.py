import os
import time
import streamlit as st
import openai
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from io import BytesIO
from tempfile import NamedTemporaryFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


# Initialize OpenAI client
client = openai.OpenAI()

# Streamlit page config
st.set_page_config(page_title="RAG-based QA App", layout="wide")
# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
# Title and description
#st.title("Enhancing Contextual Response Generation for Generative AI with Retrieval-Augmented Large Language Models using LangChain")
st.markdown("<h2 style='text-align: center;'>Enhancing Contextual Response Generation for Generative AI with Retrieval-Augmented Large Language Models using LangChain</h3>", unsafe_allow_html=True)

#st.subheader("Retrieval-Augmented Generation (RAG) App")
#st.write("Ask a question, and the model will retrieve relevant context before generating an answer.")
# Sidebar: Model Selection
# Sidebar: Domain Selection (Ensure there's only one instance)
st.sidebar.header("Domain Selection")
domain = st.sidebar.selectbox("Select Domain:", ["All", "Medical", "Legal", "Finance", "Technology"], key="domain_selection")

# Display domain-specific text based on selection
st.subheader(f"{domain} Retrieval-Augmented Generation (RAG) App")
st.write(f"Ask a {domain.lower()} question, and the model will retrieve relevant context before generating an answer.")

# Define data sources for each domain
data_sources = {
    "All": "https://commoncrawl.org/",
    "Medical": "https://www.who.int/",
    "Legal": "https://www.courtlistener.com/api/",
    "Finance": "https://www.sec.gov/edgar.shtml",
    "Technology": "https://arxiv.org/list/cs.AI/recent"
}

# Function to fetch documents (Placeholder function)
#def fetch_documents(source):
#    return [f"Data from {source}"]  # Replace with actual fetching logic

# Function to fetch real documents
def fetch_documents(source):
    try:
        response = requests.get(source, timeout=10)  # 10-second timeout
        response.raise_for_status()  # Raise error if request fails
        return [f"Data from {source}", response.text[:1000]] # Replace with actual fetching logic  # Return first 1000 characters
    except requests.exceptions.RequestException as e:
        return [f"Error fetching data from {source}: {e}"] 
       

# Fetch documents from the selected domain
data_source = data_sources.get(domain, "https://commoncrawl.org/")
docs = fetch_documents(data_source)

# Display fetched data (debugging)
print("Fetched Documents:", docs)



# Fetch documents from the selected domain
data_source = data_sources.get(domain, "https://commoncrawl.org/")
docs = fetch_documents(data_source)

# Handle case where no articles are found
if not docs:
    st.error("No articles fetched. Check the data source.")
    st.stop()

# Display fetched documents
st.write("Fetched documents:")
st.write(docs)

# Sidebar: Model Selection
st.sidebar.header("LLM Model Selection")
base_model = st.sidebar.selectbox("Select Base LLM Model:", ["OpenAI", "Hugging Face", "LLaMA", "Google Gemini", "DeepSeek-R1", "DeepSeek-V3"], key="base_model_selection")
fine_tuned_model = st.sidebar.selectbox("Select Fine-Tuned LLM Model:", ["GPT-4 Turbo", "BioBERT", "DeepSeek", "LLaMA-2", "Gemini-Pro", "DeepSeek-R1", "DeepSeek-V3"], key="fine_tuned_model_selection")
embedding_model_name = st.sidebar.selectbox("Select Embedding Model:", ["all-MiniLM-L6-v2", "msmarco-distilbert-base-v3", "multi-qa-MiniLM-L6-cos-v1", "nomic-ai/nomic-embed-text-v1"], key="embedding_model_selection")

# Define model mappings
model_mapping = {
    "GPT-4 Turbo": "gpt-4-turbo",
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "DeepSeek-V3": "deepseek-ai/deepseek-llm-7b",
    "LLaMA": "meta-llama/Llama-2-7b-chat-hf",
    "Google Gemini": "google/gemini-pro",
    "DeepSeek-R1": "deepseek-ai/deepseek-llm-7b",
    "OpenAI": "gpt-4-turbo",
    "Hugging Face": "sentence-transformers/all-MiniLM-L6-v2"
}

# Select model based on user selection (model selection logic:)
if fine_tuned_model in model_mapping:
    model_name = model_mapping[fine_tuned_model]
elif base_model in model_mapping:
    model_name = model_mapping[base_model]
else:
    model_name = "gpt-4-turbo"  # Default fallback

#model_name = model_mapping.get(fine_tuned_model, base_model)

# Display selected model
st.write("Selected Model:", model_name)

# Sidebar: Temperature Selection
st.sidebar.header("Fine-Tuning Parameters")
temperature = st.sidebar.slider("Select Temperature (Randomness)", 0.0, 1.5, 0.7, step=0.1)

# Sidebar: Input Parameters for Text Splitting
st.sidebar.header("Text Splitting Parameters")
chunk_size = st.sidebar.slider("Select Chunk Size", min_value=100, max_value=2000, value=500, step=100)
chunk_overlap = st.sidebar.slider("Select Chunk Overlap", min_value=0, max_value=500, value=50, step=10)

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

#text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
documents = text_splitter.create_documents(docs)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model_name}")
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever()
st.write("Vector Store Retriever initialized successfully")


# File Upload Section
st.sidebar.header("Upload Documents for RAG Processing")
uploaded_files = st.sidebar.file_uploader("Upload documents (PDF, DOCX, TXT,.CSV, or EXCEL)", type=["pdf", "docx", "txt",".csv","xlsx"], accept_multiple_files=True)


# Function to process multiple uploaded files
def process_uploaded_files(uploaded_files):
    all_documents = []

    if not uploaded_files:
        return None

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1]

        with NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_filepath = temp_file.name  # Save temporary file path

        try:
            if file_extension == "pdf":
                loader = PyPDFLoader(temp_filepath)
            elif file_extension == "docx":
                loader = UnstructuredWordDocumentLoader(temp_filepath)
            elif file_extension == "txt":
                loader = TextLoader(temp_filepath, encoding="utf-8")
            elif file_extension == "csv":
                loader = CSVLoader(temp_filepath)
            elif file_extension == "xlsx":
                loader = UnstructuredExcelLoader(temp_filepath)
                     
            else:
                st.error(f"Unsupported file format: {file_extension}")
                continue  # Skip unsupported files
            
            documents = loader.load()
            all_documents.extend(documents)  # Add to the list of processed docs

        except Exception as e:
            st.error(f"Error loading document {uploaded_file.name}: {str(e)}")

        #finally:
            os.remove(temp_filepath)  # Clean up temp file after processing

    return all_documents

# Process uploaded files
documents = process_uploaded_files(uploaded_files)

if documents:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded and processed successfully.")
else:
    st.sidebar.warning("No documents uploaded. You can still enter a question manually.")

if documents:
    st.sidebar.success("File uploaded and processed successfully.")
else:
    st.sidebar.warning("No document uploaded. You can still enter a question manually.")

# Text Splitting for RAG Processing
if documents:
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    #print(f"Number of split documents: {len(split_docs)}")
    st.sidebar.write(f"Number of split documents: {len(split_docs)}")
else:
    split_docs = []

#---------------------------
# Convert to FAISS Vector Store
#vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
#vectorstore.save_local("faiss_index")

#print("FAISS index created successfully!")
#-----------------------------------------------


#--------------------
print(f"Number of split documents: {len(split_docs)}")
if len(split_docs) > 0:
    print(f"First document: {split_docs[0]}")
else:
    print("No documents found after splitting!")
#--------------------------------------------------
#--------------------------------------------------
embeddings = embedding_model.embed_documents([doc.page_content for doc in split_docs])
st.sidebar.write(f"Number of embeddings generated: {len(embeddings)}")
if embeddings:
    print(f"First embedding: {embeddings[0]}")
else:
    print("Embeddings list is empty!")
#---------------------------------------------------

#----------------------------------------------------
if len(split_docs) == 0:
    raise ValueError("No documents found for FAISS indexing!")

embeddings = embedding_model.embed_documents([doc.page_content for doc in split_docs])

if len(embeddings) == 0:
    raise ValueError("No embeddings were generated!")

vectorstore = FAISS.from_documents(split_docs, embedding_model)

#--------------------------------------------------------------------------


# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model_name}")
vectorstore = FAISS.from_documents(split_docs, embedding_model)
retriever = vectorstore.as_retriever()
# print(f"Embeddings generated: {len(embeddings)}")
st.write("Retriever initialized successfully")

# Define the retrieval-augmented prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Use the provided medical context to answer the question:\nContext: {context}\nQuestion: {question}\nAnswer:"
)


# Answer generation function with Semantic Analysis
# Fix: Ensure generate_answer returns 3 values
def generate_answer(query, retriever):
    start_time = time.time()
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs[:3]]) if retrieved_docs else ""

    # Ensure correct model selection
    if fine_tuned_model in model_mapping:
        selected_model = model_mapping[fine_tuned_model]
    elif base_model in model_mapping:
        selected_model = model_mapping[base_model]
    else:
        selected_model = "gpt-4-turbo"  # Default fallback

    selected_model = model_mapping.get(fine_tuned_model, model_mapping.get(base_model, "gpt-4-turbo"))

    if selected_model.startswith("gpt-4"):
        response = client.chat.completions.create(
            model=selected_model,
            temperature=temperature,  # 🔹 Fine-tune randomness
            messages=[
                {"role": "system", "content": "You are a medical assistant providing evidence-based answers."},
                {"role": "user", "content": prompt_template.format(context=context, question=query)}
            ]
        )
        answer = response.choices[0].message.content

    elif selected_model in ["dmis-lab/biobert-base-cased-v1.1", "deepseek-ai/deepseek-llm-7b", "meta-llama/Llama-2-7b-chat-hf"]:
        medical_qa_pipeline = pipeline("text-generation", model=selected_model)
        answer = medical_qa_pipeline(f"Context: {context}\nQuestion: {query}", max_length=300,temperature=temperature)[0]['generated_text']
    else:
        answer = "Error: Selected model is not supported for this query."

    # Fix: Compute semantic similarity
    semantic_score = compute_semantic_similarity(context, answer)

    return answer, time.time() - start_time, semantic_score  # ✅ Now returns 3 values
    

# Evaluation Metrics Functions
def calculate_bleu(reference, hypothesis):
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=SmoothingFunction().method1)

def calculate_rouge(reference, hypothesis):
    return rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True).score(reference, hypothesis)

def calculate_perplexity(sentence, model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    loss = model(input_ids, labels=input_ids).loss
    return torch.exp(loss).item()

# Hallucination Check Function 
def check_hallucination(response, retrieved_context):
    overlap = sum(1 for word in response.split() if word in retrieved_context)
    hallucination_score = 1 - (overlap / len(response.split()))
    return hallucination_score

# Hallucination Scoring Functions
def fact_overlap_score(context, answer):
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())
    return len(context_words & answer_words) / len(answer_words) if answer_words else 0

# Define context variable
retrieved_docs = []
context = "\n".join([doc.page_content for doc in retrieved_docs[:3]]) if retrieved_docs else ""

# Ensure response is defined
response = ""

overlap_score = fact_overlap_score(context, response)

def check_contradiction(context, answer):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    
    inputs = tokenizer(context, answer, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    
    contradiction_score = probs[0][0].item()  # Probability of contradiction
    return contradiction_score

# Function to compute semantic similarity
def compute_semantic_similarity(context, answer):
    if not context or not answer:
        return 0.0  # Return 0 if either is missing
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    context_embedding = embedding_model.embed_documents([context])[0]
    answer_embedding = embedding_model.embed_documents([answer])[0]

    similarity_score = cosine_similarity(
        np.array(context_embedding).reshape(1, -1),
        np.array(answer_embedding).reshape(1, -1)
    )[0][0]

    return round(similarity_score, 4)


 # Compute semantic similarity
    semantic_score = compute_semantic_similarity(context, answer)

    return answer, time.time() - start_time, semantic_score

query = st.text_input("Please Enter your question ? :")
if st.button("Generate Answer"):
    if query.strip():
        with st.spinner("Please Enter Generating response..."):
            response, proc_time, semantic_score = generate_answer(query, retriever)  # ✅ Now correctly receives 3 values
            st.success("Answer:")
            st.write(response)
            st.sidebar.write(f"Processing Time: {proc_time:.2f} sec")
            st.sidebar.write(f"Temperature: {temperature:.2f}")  # 🔹 Display in sidebar
            st.sidebar.header("Semantic Analysis Score")
            st.sidebar.write(f"**Semantic Similarity: {semantic_score:.4f}**")
    else:
        st.warning("Please enter a valid question to get proper answer.")


        
    # Evaluation Metrics
    reference = "Expected medical answer based on ground truth data."
    bleu_score = calculate_bleu(reference, response)
    rouge_scores = calculate_rouge(reference, response)
    perplexity = calculate_perplexity(response)
        
    # Display Performance Metrics
    st.sidebar.header("Performance Metrics")
    metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
    values = [
        bleu_score,
        rouge_scores['rouge1'].fmeasure,
        rouge_scores['rouge2'].fmeasure,
        rouge_scores['rougeL'].fmeasure
    ]
        
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
    ax.set_ylabel("Score")
    ax.set_title("Performance Metrics")
    st.sidebar.pyplot(fig)
        
    # Separate Perplexity Graph
    fig_perp, ax_perp = plt.subplots(figsize=(5, 3))
    ax_perp.bar(["Perplexity"], [perplexity], color=['orange'])
    ax_perp.set_ylabel("Score")
    ax_perp.set_title("Perplexity Score")
    st.sidebar.pyplot(fig_perp)
        
    # Hallucination Check
    overlap_score = fact_overlap_score(context, response)
    contradiction_score = check_contradiction(context, response)

    # Display Hallucination Metrics

    # Format the numbers to display with more decimal places (e.g., 6 decimal places)
   #formatted_fact_overlap = f"{fact_overlap_score:.6f}"
    #formatted_contradiction = f"{contradiction_score:.6f}"

#print(f"Hallucination Check\nFact Overlap Score: {formatted_fact_overlap}\nContradiction Score: {formatted_contradiction}")
    st.sidebar.header("Hallucination Check")
    st.sidebar.write(f"Fact Overlap Score: {overlap_score:.6f}")
    st.sidebar.write(f"Contradiction Score: {contradiction_score:.6f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Fact Overlap", "Contradiction"], [overlap_score, contradiction_score], color=['green', 'red'])
    ax.set_ylabel("Score")
    ax.set_title("Hallucination Metrics")
    st.sidebar.pyplot(fig)

    st.sidebar.write(f"Processing Time: {proc_time:.2f} sec")
else:
    st.warning("Please enter a valid question.") 





   