#PdfReader:pypdf库里的专门读取pdf文档的工具
#它能打开pdf文件，提取里面的文字也只能提取文字
#TfidfVectorizer：把纯文本字符串，转换成向量数字，计算机才能依此计算文本相似度
#传入一堆文本chunk输出TF-IDF向量矩阵
#cosine_similarity:计算(问题的向量)和(文档块的向量)两端文本语义的相似度
#返回值介于0-1，越接近1越相关，计算机也是根据这个选出最相关的文档片段
#numpy：py数值计算库，处理矩阵和数组
#TF-IDF输出的向量是矩阵，排序取Top相关片段全靠numpy操作

#核心变量：
#doc_text:读取出的文档全部长文本(一大段长字符串)
#chunks：被切出来的小块文本列表
#vectorizer：TfidfVectorizer的实例文本向量化器
#chunk_vectors：所有chunks转化成的TIF-IDF向量矩阵
#query：用户提问的字符串
#query_vector：用户提问转出来的向量
#similarities：query和每一个chunk之间的余弦相似度分数数组
#top_indices：相似度最高的几个chunk的下标编号，用来取出对应的文档片段
#top_k = x:默认返回相似度最高的前x段文本

import os
from operator import truediv

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key =os.getenv("DEEPSEEK_API_KEY"),base_url = "https://api.deepseek.com" )

#加载文档函数：
def load_document(path):
    if path.endswith(".pdf"):
#判断文档后缀是否为“pdf”
        reader = PdfReader(path)
        text = ""
#定义空字符串用来存放全部取出来的文件
        for page in reader.pages:
            text += page.extract_text()
#提取并拼接
        return text
    else:
        with open(path, 'r',encoding = 'utf-8') as f:
#with：当代码跑出with缩进块自动关闭文件，不容易出bug
#‘r’只读取内容不做修改
#encoding = 'utf-8':使用utf-8这套规则读取文件
#as f：命名为f
            return f.read()
#f.read()一次性读取全部文字内容，返回一整个长字符串

#文档切块：
def split_into_chunks(text,chunk_size = 350):
    chunks = []
    for i in range (0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

#构建检索向量函数：
#训练TF-IDF向量器
def build_retriever (chunks):
    vectorizer =    TfidfVectorizer()
#将TfidfVectorizer实例化
    chunk_vectors = vectorizer.fit_transform(chunks)
#.fit(chunks):通读文档块并收集，建立词表，统计出现频率
#.transform(chunks):将.fit(chunks)学到的词表转变为数字向量
#文档库用“fit_transform”既要学习词表又要转数字向量
#用户提问用“transform”不能用fit，加上fit会重新学习丢失相似度
#.fit_transform：先学习文档里的单词，再将其转为数字向量存到chunk_vectors
    return vectorizer,chunk_vectors

#相似度检索函数：
def retrieve(query,chunks,vectorizer,chunk_vectors,top_k = 3):
    query_vectors = vectorizer.transform([query])
#用户提问用“transform”不能用fit，加上fit会重新学习丢失相似度
    similarities = cosine_similarity(query_vectors,chunk_vectors)[0]
#cosine_similarities(a,b):余弦相似度，计算a,b之间的相似度
#输出是一个相似度数组数字越接近1，代表两文档越相关
    top_indices = np.argsort(similarities)[-top_k:][::-1]
#argsort:从小到大排序返回值是下标非数值
#[-top_k:]：取最后面top_k个(分数最高的几个的下标)
#[::-1]:数组反转变为从高到低排序
    return[chunks[i] for i in top_indices]

def generate_answer (query,retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
#\n\n:片段和片段之间空一行隔开，llm更容易区分不同段落
#.join():把列表中全部字符串拼接成一整个长字符串
    prompt = f"""
你是一个回答助手。请严格根据下面提供的参考资料回答问题。
若资料中找不到相关信息，请直接回答“根据提供的信息无法回答该问题”。
【参考资料】：
{context}
【问题】：
{query}
"""
#prompt:prompt约束用来解决llm幻觉防止它乱说话
    response = client.chat.completions.create(
        model = "deepseek-chat",
        messages = [{"role":"user","content":prompt}],
        temperature = 0
    )
    return response.choices[0].message.content
#注意这里用的是单数的message，而之前的是复数messages
#单数的message：表示ai给出的单句回答
#复数的messages：表示1传入给ai的一大串字符串，像问题，要求之类的
#llm返回值是一个json结构体
#choice[0]取第一条回答结果
#.message.content:取出模型生成的文章答案返回给调用者

