# PdfReader:pypdf库里的专门读取pdf文档的工具
# 它能打开pdf文件，提取里面的文字也只能提取文字
# TfidfVectorizer：把纯文本字符串，转换成向量数字，计算机才能依此计算文本相似度
# 传入一堆文本chunk输出TF-IDF向量矩阵
# cosine_similarity:计算(问题的向量)和(文档块的向量)两端文本语义的相似度
# 返回值介于0-1，越接近1越相关，计算机也是根据这个选出最相关的文档片段
# numpy：py数值计算库，处理矩阵和数组
# TF-IDF输出的向量是矩阵，排序取Top相关片段全靠numpy操作

# 核心变量：
# doc_text:读取出的文档全部长文本(一大段长字符串)
# chunks：被切出来的小块文本列表
# vectorizer：TfidfVectorizer的实例文本向量化器
# chunk_vectors：所有chunks转化成的TIF-IDF向量矩阵
# query：用户提问的字符串
# query_vector：用户提问转出来的向量
# similarities：query和每一个chunk之间的余弦相似度分数数组
# top_indices：相似度最高的几个chunk的下标编号，用来取出对应的文档片段
# top_k = x:默认返回相似度最高的前x段文本

import os
from operator import truediv

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

# SentenceTransformer：专门用来将一句话/一段文本，输出一个固定长度向量
embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
# ‘BAAI’：北京人工智能研究院
# bge：双语通用向量模型
# small：小体积版本，速度快，吃显存少本地电脑容易跑起来
# zh：专门优化中文
# v1.5：版本号
# embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')此次升级的核心
# 能将pdf文档或者是用户问题经embed_model.encode()处理转化成代表‘意思’的向量
import chromadb
# chromadb：轻量级开源向量数据库，能将文档切片和embedding向量存起来，需要时快速找出语义最相近的片段交给llm
# 相当于一个存放文档切片和embedding向量的仓库
# import pymupdf
import fitz
import re
from dotenv import load_dotenv
from openai import OpenAI
import jieba
from rank_bm25 import BM25Okapi
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


# 加载文档函数：
def load_document(path):
    if path.endswith(".pdf"):
        # 判断文档后缀是否为“pdf”
        import pymupdf
        doc = pymupdf.open(path)
        # 打开文件并粘贴给doc
        cleaned_text = ""
        raw_text = ""
        for i, page in enumerate(doc):
            raw_text = page.get_text()
            print(f"正在清洗{i + 1}页...")
            clean_text = clean_page_with_llm(raw_text)

            match = re.search(
                r"(?:打印日期|印刷日期)\s*[:: ]?\s*(\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}\s*日?)",
                        raw_text,
            )
            if match:
                data_value = re.sub(r"\s+","",match.group(1))
                clean_text += f"\n打印日期：{data_value}"

            cleaned_text += clean_text + "\n\n"
        doc.close()
        return cleaned_text
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


# cleaned_text:最终清洗的全文
# raw_text；原始文本
# clean_text:经由llm清洗得到的文本块

# 文档切块：
def split_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


# 构建检索向量函数：
# 训练TF-IDF向量器
def build_retriever(chunks):
    chroma_client = chromadb.Client()
    # 创建一个chroma仓库
    collection = chroma_client.get_or_create_collection(name="rag_docs")
    # 到chroma仓库中找是否有叫‘rag_docs’的集合(collection),没有就建一个
    # 此处可以理解为chroma_client相当于一个大仓库，collection相当于一个柜子，柜子名为rag_docs
    embeddings = embed_model.encode(chunks).tolist()
    # 一次性将全部的chunks文本批量交给bge模型，全部转化为向量数组
    # chunks：字符串列表
    # .encode():批量编码一次性把列表里的文字都算出对应的embedding向量
    # .tolist():将numpy转为py原生数组
    # chroma不接受numpy数组只能用.tolist()转为原生数组后存入chroma当中
    collection.add(
        documents=chunks,
        embeddings=embeddings,  # 后面的embedding=chunk_vectors
        ids=[str(i) for i in range(len(chunks))]
    )
    # document：存的是原始文字，当检索匹配上以后chroma可以返回原始文字拿去拼prompt给llm
    # embeddings:接收来自‘embeddings = embed_model.encode(chunks).tolist()’的向量
    # 后面的embedding可以改成chunk_vectors更好理解
    # str():转string类型
    # str(i) for i in range(len(chunks))：生成0-len(chunks)-1个数字
    # chroma的id只接收字符串，直接传数字会报错所以要转一下
    # 这里是为了个每一段字符串加一个下标

    tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]
    bm25_model = BM25Okapi(tokenized_chunks)

    return collection, bm25_model, chunks


# 相似度检索函数：
def retrieve(query, collection, bm25_model, chunks, top_k=3):  # <-这里定义了返回最相似的前3个结果
    query_vector = embed_model.encode([query]).tolist()
    vector_results = collection.query(query_embeddings=query_vector, n_results=top_k * 2)
    vector_doc = vector_results['documents'][0]

    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25_model.get_scores(tokenized_query)
    bm25_scores = np.array(bm25_scores).ravel()
    bm25_top_indices = bm25_scores.argsort()[-top_k * 2:][::-1]
    valid_indices = [i for i in bm25_top_indices if i < len(chunks)]
    bm25_docs = [chunks[i] for i in valid_indices]

    rrf_scores = {}
    for rank, doc in enumerate(vector_doc):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (60 + rank)
    for rank, doc in enumerate(bm25_docs):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (60 + rank)

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final_docs = [doc for doc, score in sorted_docs[:top_k]]

    date_terms = ("打印日期","印刷日期")
    if any(term in query for term in date_terms):
        date_chunks = [
            chunk for chunk in chunks
            if any (term in chunk for term in date_terms)
        ]
        final_docs = (date_chunks + [chunk for chunk in final_docs if chunk not in date_chunks])[:top_k]

    print(f"\n混合检索：{query}->最终选出的文本块是：\n{final_docs[0][:150]}...\n")
    return final_docs


# 此处同理
def clean_page_with_llm(page_text):
    """
    调用llm，把单页乱码文本清洗成Markdown表格
    """
    if len(page_text.strip()) < 50:
        return page_text

    if len(page_text) > 4000:
        print("页面文本过长，跳过大模型清洗")
        return page_text

    prompt = f"""你是一个专业的数据清洗助手。下面是一页从 PDF 里提取出来的原始文字，排版极度混乱，尤其是表格内容错位。
请你帮我做两件事：
1. 修复段落结构，保持原意不变。
2. 如果发现表格，请严格把它转化为标准的 Markdown 表格格式（用 | 和 - 分隔）。
【绝对禁止】自己编造数据，只能整理和重排原文出现的文字和数字。
【特别注意】必须保留页眉、页脚、页码、打印日期等边缘信息，别把它们当做噪音丢弃！
【原始文字】：
{page_text}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"清洗页面失败，跳过这页的清洗：{e}")
        return page_text

    if len(page_text) > 4000:
        return page_text


def generate_answer(query, retrieved_chunks):

    if any(term in query for term in ("打印日期","印刷日期")):
        for chunk in retrieved_chunks:
            chunk_text = chunk[0] if isinstance(chunk, list) else chunk
            normalized = re.sub(r"\s+","",chunk)
            match = re.search(
                r"(?:打印日期|印刷日期)[:: ]?(\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)",
            normalized
            )
            if match:
                return f"文件的印刷日期是：{match.group(1)}"

    context = "\n\n".join(retrieved_chunks)
    # \n\n:片段和片段之间空一行隔开，llm更容易区分不同段落
    # .join():把列表中全部字符串拼接成一整个长字符串
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
    # prompt:prompt约束用来解决llm幻觉防止它乱说话
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
# 注意这里用的是单数的message，而之前的是复数messages
# 单数的message：表示ai给出的单句回答
# 复数的messages：表示1传入给ai的一大串字符串，像问题，要求之类的
# llm返回值是一个json结构体
# choice[0]取第一条回答结果
# .message.content:取出模型生成的文章答案返回给调用者


