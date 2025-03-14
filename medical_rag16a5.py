#Python Library Loading
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from transformers import pipeline
import bert_score



import os
import time
import streamlit as st
import openai
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
import pinecone
import faiss
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tempfile import NamedTemporaryFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# Initialize OpenAI client
client = openai.OpenAI()

# Streamlit page config
st.set_page_config(page_title="RAG-based QA App", layout="wide")

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY=os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_GEMINI_API_KEY=os.getenv("GOOGLE_GEMINI_API_KEY")
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
LLAMA_API_KEY=os.getenv("LLAMA_API_KEY")                
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
PINCON_API_KEY=os.getenv("PINCONE_API_KEY")

# Title and description
#st.title("Enhancing Contextual Response Generation for Generative AI with Retrieval-Augmented Large Language Models using LangChain")
st.markdown("<h2 style='text-align: center;'>Enhancing Contextual Response Generation for Generative AI with Retrieval-Augmented Large Language Models using LangChain</h3>", unsafe_allow_html=True)

st.sidebar.header("Domain Selection")
domain = st.sidebar.selectbox("Select Domain:", ["All", "Medical", "Legal", "Finance", "Technology"], key="domain_selection")

# Display domain-specific text based on selection
st.subheader(f"{domain} Retrieval-Augmented Generation (RAG) App")
st.write(f"Ask a {domain.lower()} question, and the model will retrieve relevant context before generating an answer.")

# Define data sources for each domain
data_sources = {
    "All": "https://en.wikipedia.org/wiki", #"https://commoncrawl.org/", "https://www.kaggle.com/datasets", "https://www.data.gov/", "https://www.data.gov.in/",
    "Medical": "https://www.who.int/", #"https://www.pubmed.ncbi.nlm.nih.gov/"],
    "Legal": "https://www.freelaw.in/",#"https://www.legalserviceindia.com/"],
    "Finance": "https://www.rbi.org.in/Scripts/Statistics.aspx", #"https://www.bseindia.com/"],
    "Technology": "https://arxiv.org/list/cs.AI/recent", #"https://www.techradar.com/"]
}

# Function to fetch real documents (Placeholder function)
def fetch_documents(source):
    try:
        response = requests.get(source, timeout=10)  # 10-second timeout
        response.raise_for_status()  # Raise error if request fails
        return [f"Data from {source}"]  # Replace with actual fetching logic
        #return [f"Data from {source}", response.text[:500]] # Replace with actual fetching logic  # Return first 1000 characters
    except requests.exceptions.RequestException as e:
        return [f"Error fetching data from {source}: {e}"] 
       

# Fetch documents from the selected domain
data_source = data_sources.get(domain, "https://en.wikipedia.org/wiki")
docs = fetch_documents(data_source)

# Display fetched data (debugging)
print("Fetched Documents:", docs)

# Handle case where no articles are found
if not docs:
    st.error("No articles fetched. Check the data source.")
    st.stop()

#st.write("### 🌍 Fetched Webpage Preview")
#st.components.v1.iframe(data_source, height=600, scrolling=True)

# Display fetched documents
st.write("Fetched documents for Fine Tunning LLM :")
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
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
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

model_name = model_mapping.get(fine_tuned_model, base_model)

# Display selected model
st.write("Selected Base LLM Model:", base_model)
st.write("Selected Fine-Tunned LLM Model:", fine_tuned_model)
st.write("Selected Embedding Model:", embedding_model_name)
st.write("Selected Model:", model_name)


# Sidebar: Temperature Selection
st.sidebar.header("Fine-Tuning Parameters")
temperature = st.sidebar.slider("Select Temperature (Higher=more randomness, Lower=more focused)", 0.0, 1.5, 0.7, step=0.1)

# Sidebar: Input Parameters for Text Splitting
st.sidebar.header("Text Splitting Parameters")
chunk_size = st.sidebar.slider("Select Chunk Size", min_value=100, max_value=2000, value=500, step=100)
chunk_overlap = st.sidebar.slider("Select Chunk Overlap", min_value=0, max_value=500, value=50, step=10)

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

#text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
documents = text_splitter.create_documents(docs)


# File Upload Section
st.sidebar.header("Upload Documents for RAG Processing")
uploaded_files = st.sidebar.file_uploader("Upload documents (PDF, DOCX, TXT,.CSV, or EXCEL)", type=["pdf", "docx", "txt",".csv","xlsx"], accept_multiple_files=True)

################# Load Documents if Uploaded
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded.")

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
    st.sidebar.warning("No document uploaded. You can still enter a question manually, Using LLM for direct responses..")


# Text Splitting for RAG Processing
if documents:
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    #print(f"Number of split documents: {len(split_docs)}")
    st.sidebar.write(f"Number of split documents: {len(split_docs)}")
else:
    split_docs = []



# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model_name}")

if documents:
    vectorstore = FAISS.from_documents(documents, embedding_model)
    retriever = vectorstore.as_retriever()
    st.write("Vector Store Retriever initialized successfully")
else:
    st.error("No documents available for creating the vector store.")

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
    if not documents:
        print("⚠️ No documents found. Skipping FAISS indexing.")

embeddings = embedding_model.embed_documents([doc.page_content for doc in split_docs])


if not split_docs or not embeddings:
    print("⚠️ No valid documents or texts found. Skipping FAISS indexing.")
    vectorstore = None
    
else:
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    st.sidebar.write("✅ FAISS index created successfully!")

   
if len(embeddings) == 0:
    raise ValueError("Please upload documents for LLM Fine Tunning.") #No embeddings were generated!

#vectorstore = FAISS.from_documents(split_docs, embedding_model)


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
                {"role": "system", "content": "You are a domain assistant providing evidence-based answers."},
                {"role": "user", "content": prompt_template.format(context=context, question=query)}
            ]
        )
        answer = response.choices[0].message.content

    elif selected_model in ["gpt-4-turbo","dmis-lab/biobert-base-cased-v1.1", "deepseek-ai/deepseek-llm-7b", "meta-llama/Llama-2-7b-chat-hf"]:
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

from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from transformers import pipeline
import bert_score

# Load pre-trained models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence Embeddings
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")  # NLI Model

def check_hallucination_advanced(response, retrieved_contexts, sim_threshold=0.5, nli_threshold=0.7):
    """
    Detect hallucination using Cosine Similarity, BERTScore, and NLI.
    
    Parameters:
    - response (str): Model-generated response
    - retrieved_contexts (list of str): List of retrieved supporting facts
    - sim_threshold (float): Cosine similarity threshold
    - nli_threshold (float): Minimum probability for 'Entailment' to accept response

    Returns:
    - is_hallucination (bool): True if the response is likely hallucinated
    - similarity_score (float): Max cosine similarity score
    - entailment_score (float): Max entailment probability from NLI
    - bert_score (float): Maximum semantic similarity from BERTScore
    """

    # Step 1: Compute Cosine Similarity
    response_embedding = embedding_model.encode(response, convert_to_tensor=True)
    context_embeddings = embedding_model.encode(retrieved_contexts, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(response_embedding, context_embeddings)
    max_similarity = np.max(similarity_scores.numpy())

    # Step 2: Compute BERTScore for each reference separately
    bert_scores = []
    for ref in retrieved_contexts:
        P, R, F1 = bert_score.score([response], [ref], lang="en", rescale_with_baseline=True)
        bert_scores.append(float(F1[0]))  # Extract scalar value
    max_bert_score = max(bert_scores) if bert_scores else 0.0  # Avoid empty list error

    # Step 3: Natural Language Inference (NLI)
    entailment_scores = []
    for context in retrieved_contexts:
        result = nli_model(f"Hypothesis: {response} \nPremise: {context}")
        entailment_prob = next((item['score'] for item in result if item['label'] == 'ENTAILMENT'), 0)
        entailment_scores.append(entailment_prob)
    
    max_entailment = max(entailment_scores) if entailment_scores else 0.0

    # Step 4: Flag Hallucination based on all criteria
    is_hallucination = max_similarity < sim_threshold and max_entailment < nli_threshold and max_bert_score < 0.7

    return is_hallucination, max_similarity, max_entailment, max_bert_score

# Example Usage
response = "The Eiffel Tower is located in Berlin."
retrieved_contexts = [
    "The Eiffel Tower is a famous landmark in Paris, France.",
    "It was constructed in 1889 and is one of the most visited monuments in the world."
]

hallucination_flag, similarity, entailment, bert_score_val = check_hallucination_advanced(response, retrieved_contexts)

print(f"Hallucination: {hallucination_flag}")
print(f"Cosine Similarity: {similarity:.4f}")
print(f"Entailment Score: {entailment:.4f}")
print(f"BERTScore: {bert_score_val:.4f}")
print(f"Hallucination: {hallucination_flag}, Similarity: {similarity}, Entailment: {entailment}, BERTScore: {bert_score_val}")


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

    st.sidebar.header("Hallucination Check Metrices")
    st.sidebar.write(f"Hallucination: {hallucination_flag}")
    st.sidebar.write(f"Entailment: {entailment:.6f}")
    st.sidebar.write(f"Cosine Similarity: {similarity:.6f}")   
    st.sidebar.write(f"BERTScore: {bert_score_val:.6f}")                  
    st.sidebar.write(f"Fact Overlap Score: {overlap_score:.6f}")
    st.sidebar.write(f"Contradiction Score: {contradiction_score:.6f}")


    # Visualization - Cosine & BERT
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Cosine Similarity","BERTScore"], [similarity,bert_score_val], color=['green', 'red',])
    ax.set_ylabel("Score")
    ax.set_title("Hallucination Metrics")
    st.sidebar.pyplot(fig)

# Visualization - Fact Overlap  & Contradiction, Entailment
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Fact Overlap", "Contradiction","Entailment"], [overlap_score, contradiction_score,entailment], color=['blue','purple','orange'])
    ax.set_ylabel("Score")
    st.sidebar.pyplot(fig)


    #st.sidebar.write(f"Processing Time: {proc_time:.2f} sec")
else:
    st.warning("Please enter a valid question.") 





   