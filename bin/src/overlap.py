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
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
#SentenceTransformer：专门用来将一句话/一段文本，输出一个固定长度向量
embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
#‘BAAI’：北京人工智能研究院
#bge：双语通用向量模型
#small：小体积版本，速度快，吃显存少本地电脑容易跑起来
#zh：专门优化中文
#v1.5：版本号
#embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')此次升级的核心
#能将pdf文档或者是用户问题经embed_model.encode()处理转化成代表‘意思’的向量
import chromadb
#chromadb：轻量级开源向量数据库，能将文档切片和embedding向量存起来，需要时快速找出语义最相近的片段交给llm
#相当于一个存放文档切片和embedding向量的仓库
#import pymupdf
import fitz
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key =os.getenv("DEEPSEEK_API_KEY"),base_url = "https://api.deepseek.com" )

#加载文档函数：
def load_document(path):
    if path.endswith(".pdf"):
#判断文档后缀是否为“pdf”
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
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
def split_into_chunks(text,chunk_size = 500,overlap = 100):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start +=(chunk_size - overlap)
    return chunks

#构建检索向量函数：
#训练TF-IDF向量器
def build_retriever (chunks):
    chroma_client = chromadb.Client()
#创建一个chroma仓库
    collection = chroma_client.get_or_create_collection(name="rag_docs")
#到chroma仓库中找是否有叫‘rag_docs’的集合(collection),没有就建一个
#此处可以理解为chroma_client相当于一个大仓库，collection相当于一个柜子，柜子名为rag_docs
    embeddings = embed_model.encode(chunks).tolist()
#一次性将全部的chunks文本批量交给bge模型，全部转化为向量数组
#chunks：字符串列表
#.encode():批量编码一次性把列表里的文字都算出对应的embedding向量
#.tolist():将numpy转为py原生数组
#chroma不接受numpy数组只能用.tolist()转为原生数组后存入chroma当中
    collection.add(
        documents=chunks,
        embeddings=embeddings,#后面的embedding=chunk_vectors
    ids=[str(i) for i in range(len(chunks))]
    )
#document：存的是原始文字，当检索匹配上以后chroma可以返回原始文字拿去拼prompt给llm
#embeddings:接收来自‘embeddings = embed_model.encode(chunks).tolist()’的向量
#后面的embedding可以改成chunk_vectors更好理解
#str():转string类型
#str(i) for i in range(len(chunks))：生成0-len(chunks)-1个数字
#chroma的id只接收字符串，直接传数字会报错所以要转一下
#这里是为了个每一段字符串加一个下标
    return collection

#相似度检索函数：
def retrieve(query,collection,top_k = 3):   #<-这里定义了返回最相似的前3个结果
    query_vector =embed_model.encode([query]).tolist()
#[query]:本身是单个字符串加了'[]'包装成只有一个元素的字符串列表
#这里是将单个的字符串列表[query]交给bge模型，转化成向量
# .tolist():将numpy转为py原生数组
# chroma不接受numpy数组只能用.tolist()转为原生数组后存入chroma当中
    results = collection.query(
        query_embeddings = query_vector,
        n_results = top_k,
    )
#query_embeddings = query_vector：接收刚刚生成好的问题向量
#n_results：chroma库query查询接口的固定参数
#作用：让向量库返回多少条最相似的结果
#余弦相似度不用写chroma内部会自动把全部向量进行计算与排序，选出最匹配的片段
    print(f"\n查找：{query} ->搜出的文本块是：\n{results['documents'][0][0][:100]}...\n")
#chroma：返回的results是个字典，是个多层嵌套的列表
#results['documents']：是双层嵌套列表，故有两层[]
#第一个[0]:第一个问题的检索结果
#第二个[0]：检索结果里排第一的
#[:100]:文本可能很长，只截取前100个字符
    return results['documents'][0]
#此处同理

def generate_answer (query,retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
#\n\n:片段和片段之间空一行隔开，llm更容易区分不同段落
#.join():把列表中全部字符串拼接成一整个长字符串
    prompt = f"""
你是一个严格的内容提取助手。请**完全依据**下面提供的[参考资料]回答问题。
[重要规则]：
1.你是一个“检索提取器”，不是“计算器”。不要自己编写公式、进行计算、或者推演逻辑。
2.如果问题是问某个数值（比如“必修多少学分”），请优先在资料中寻找“毕业总学分”、“学分修要求”这类小节标题后面的详细列表。找到了就直接原样提取出来
3.如果资料里给出的是百分比（例如86.13%），请不要试图用总学分去乘这个百分比，直接原样提取

【参考资料】：
{context}
【问题】：
{query}
【回答】：
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

