import os
import re
from dotenv import load_dotenv
import jieba
from openai import OpenAI
from rank_bm25 import BM25Okapi
import chromadb
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
import numpy as np
import pymupdf

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url = "https://api.deepseek.com")
embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

def load_document (path):
    if path.endswith(".pdf"):
        with pymupdf.open(path) as doc:
            cleaned_text =""
            for i,line in enumerate(doc):
                raw_text =line.get_text()
                print(f"正在打印第{i+1}页...")
                cleaned_text += clean_with_llm(raw_text) + "\n\n"
        return cleaned_text
    else:
        with open(path,'r',encoding="utf-8") as f:
            return f.read()

def split_into_chunks(text,chunk_size=500,overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def build_retriever(chunks,):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name= "rag.docs")
    embeddings = embed_model.encode(chunks).tolist()
    collection.add(
        embeddings=embeddings,
        documents = chunks,
        ids = [str(i) for i in range(len(chunks))]
    )
    tokenized_chunks = [list(jieba.cut(chunk))for chunk in chunks]
    bm25_model = BM25Okapi(tokenized_chunks)
    return collection,bm25_model,chunks

def retrieve(query,collection,bm25_model,chunks,top_k=5):
    query_vector = embed_model.encode([query]).tolist()
    vector_results = collection.query(query_embeddings=query_vector,n_results = top_k*2)
    v_docs = []
    for d in vector_results['documents'][0]:
        if isinstance(d,list):
            v_docs.append(str(d[0]))
        else:
            v_docs.append(str(d))

    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25_model.get_scores(tokenized_query)
   
    bm25_top_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]
    valid_indices = [i for i in bm25_top_indices if i<len(chunks)]
    b_docs = [str(chunks[i]) for i in valid_indices]

    rrf_scores = {}
    for rank, doc in enumerate(v_docs):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (60 + rank)
    for rank, doc in enumerate(b_docs):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (60 + rank)

    sorted_docs = sorted(rrf_scores.items(),key = lambda x:x[1],reverse = True)
    final_docs = [doc for doc,score in sorted_docs[:top_k]]
    if final_docs:
        print(f"\n[混合检索]最终选出的第一块文本块是:\n{final_docs[0][:100]}...\n")
    return final_docs

def clean_with_llm(raw_text):
    if len(raw_text.strip())<50 or len(raw_text.strip())>4000:
        print("文本异常，跳过模型清洗")
        return raw_text
    prompt = f"""你是一个专业的数据清洗助手。下面是一页从 PDF 里提取出来的原始文字，排版极度混乱，尤其是表格内容错位。
    请你帮我做两件事：
    1. 修复段落结构，保持原意不变。
    2. 如果发现表格，请严格把它转化为标准的 Markdown 表格格式（用 | 和 - 分隔）。
    【绝对禁止】自己编造数据，只能整理和重排原文出现的文字和数字。
    【特别注意】必须保留页眉、页脚、页码、打印日期等边缘信息，别把它们当做噪音丢弃！
    【原始文字】：
    {raw_text}
    """
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"清洗失败：{e}")
        return raw_text

def generate_answer(query,retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    你是一个严格的内容提取助手。请**完全依据**下面提供的[参考资料]回答问题。
[重要规则]：
1. 你是一个“检索提取器”，不是“计算器”。不要自己编写公式、进行计算、或者推演逻辑。
2. 如果问题是问某个数值（比如“必修多少学分”），请优先在资料中寻找“毕业总学分”、“学分修要求”这类小节标题后面的详细列表。找到了就直接原样提取出来。
3. 如果资料里给出的是百分比（例如86.13%），请不要试图用总学分去乘这个百分比，直接原样提取。
【参考资料】：
{context}
【问题】：
{query}
【回答】：
    """
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages = [{"role":"user","content":prompt}],
        temperature = 0
    )
    return resp.choices[0].message.content
