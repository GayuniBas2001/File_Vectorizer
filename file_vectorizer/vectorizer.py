import os
import io
import re
import base64
import logging
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word files
# import pandas as pd  # For reading Excel files
import tiktoken  # Tokenizer for token length counting
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# PGSQL connection info
PG_HOST = os.getenv("PG_HOST")
PG_DB = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_PORT = os.getenv("PG_PORT", 5432)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=PG_HOST,
    database=PG_DB,
    user=PG_USER,
    password=PG_PASSWORD,
    port=PG_PORT
)
cursor = conn.cursor()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def read_pdf(pdf_b64):
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream)
        return "\n".join([page.get_text("text") for page in doc])
    except Exception as e:
        logger.error(f"PDF read error: {e}")
        return ""

# def read_word(docx_b64):
#     try:
#         word_bytes = base64.b64decode(docx_b64)
#         stream = io.BytesIO(word_bytes)
#         doc = docx.Document(stream)
#         return "\n".join([para.text for para in doc.paragraphs])
#     except Exception as e:
#         logger.error(f"Word read error: {e}")
#         return ""

# # def read_excel(xlsx_b64):
# #     try:
# #         excel_bytes = base64.b64decode(xlsx_b64)
# #         stream = io.BytesIO(excel_bytes)
# #         df = pd.read_excel(stream, sheet_name=None)
# #         return "\n\n".join([f"Sheet: {s}\n{df[s].to_string(index=False)}" for s in df])
# #     except Exception as e:
# #         logger.error(f"Excel read error: {e}")
# #         return ""

def chunk_text(text, max_tokens=384, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " "],
        length_function=lambda txt: len(tiktoken.get_encoding("cl100k_base").encode(txt))
    )
    return splitter.split_text(text)

def embed_text_chunks(chunks):
    return model.encode(chunks)

def store_embeddings(embeddings, chunks):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id SERIAL PRIMARY KEY,
            text_chunk TEXT,
            embedding public.VECTOR(384)
        )
    """)
    for emb, chunk in zip(embeddings, chunks):
        emb_list = emb.tolist()
        cursor.execute(
            "INSERT INTO document_embeddings (text_chunk, embedding) VALUES (%s, %s)",
            (chunk, emb_list)
        )
    conn.commit()

# def search_similar_chunks(query, top_k=5):
#     query_embedding = model.encode(query)
#     cursor.execute("SELECT id, text_chunk, embedding FROM document_embeddings")
#     results = cursor.fetchall()
#     scored = []
#     for _id, chunk, emb in results:
#         sim = cosine_similarity(query_embedding, np.array(emb))
#         scored.append((sim, chunk))
#     scored.sort(reverse=True)
#     return [chunk for _, chunk in scored[:top_k]]

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example Usage:
if __name__ == '__main__':

    with open("Bioinformatics_01.pdf", "rb") as f:
        b64_content = base64.b64encode(f.read()).decode('utf-8')
    raw_text = read_pdf(b64_content)
    cleaned = clean_text(raw_text)
    # print(cleaned[:1000])  # Print first 1000 characters for debugging
    chunks = chunk_text(cleaned)
    # print(chunks[:3])  # Print first 3 chunks for debugging
    embeddings = embed_text_chunks(chunks)
    print("Embeddings shape:", embeddings.shape)
    store_embeddings(embeddings, chunks)

    # top_chunks = search_similar_chunks("What is the document about?")
    # for c in top_chunks:
    #     print("-", c)
