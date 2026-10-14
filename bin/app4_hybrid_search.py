import os
import tempfile
#tempfile：临时文件
#streamlit上传的文件是存在内存里的PdfReader不能直接读，要先写到硬盘临时文件，读取完可以自动删除
import streamlit as st
#streamlit：专门快速做ai网页demo的库，不用写html/js
from src.hybrid_search import load_document,split_into_chunks,build_retriever,retrieve,generate_answer

st.set_page_config(page_title="我的rag智能问答助手",page_icon="📚")
#浏览器标签页
st.title("📚我的rag智能问答助手")
#主界面标题设置 <-相当于大标题
st.write("上传一份pdf文件，我基于它的内容回答你的问题")
#st.write():streamlit最常用的输出函数,可以输出多种东西，如字符串，列表等等 <-相当于正文
#效果：网页页面上直接显示文字

with st.sidebar:
#st.sidebar:页面侧边栏
    st.header("上传文档")
    uploaded_file =st.file_uploader("请选择你的pdf文件",type=["pdf"])
#st.file_uploader:网页上传控件，只允许pdf

    st.markdown("---")
    if st.button("清空聊天记录"):
        st.session_state.messages = []
        if "collection"  in st.session_state:
            del st.session_state.collection
        st.rerun()
#st.button():网页上渲染一个按钮，点他就执行里面的代码
#st.session_state.messages = []:将该列表置为空，清除聊天记录
#if "collection"  in st.session_state:判断内存里是否还有向量库集合
#del st.session_state.collection:删除该向量集
#st.rerun():刷新整个网页，页面回到初始状态

#动态接收上传文件
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
#tempfile.NameTemporaryFile:py内置模块，用来自动创建一个带真实磁盘路径的临时文件，用完自动删
#TemporaryFile：没有文件名拿不到磁盘路径，只能py自己读写，其他程序访问不到
#NameTemporaryFile:有真实路径name，别的库可以根据该路径打开他
#suffix=".pdf":创建出来的临时文件是以'.pdf'结尾的
#delete=False：退出with时不会自动删除，文件依旧保留在磁盘上
#一般默认值是'delete=True'表示退出with时会自动删除临时文件
#tmp_file:是NameTemporaryFile返回的文件对象，文件对象没有getvalue
#getvalue:st.file_uploader返回对象的方法

#streamlit特点:每次交互整个py文件会从头到尾重新跑一次
        if "collection"not in st.session_state:   #只在第一次上传pdf文件时构建向量库，重复上传才会重新建，避免重复建立知识库而降低速度
            with st.spinner("正在读取文档并建立语义索引..."):
#st.spinner:页面加载转圈提示
                doc_text = load_document(tmp_path)
#读取临时文件里的文字并提取出来给doc_text
                chunks = split_into_chunks(doc_text)
                st.session_state.collection,st.session_state.bm25_model,st.session_state.chunks= build_retriever(chunks)
#st.session_state:streamlit的会话状态存储，页面刷新、输入文字、点击按钮都不会丢数据
#此处是将建好的向量库collection存入st.session_state防丢
                st.success(f"知识库准备就绪，共{len(chunks)}块")
#成功提示框

if "messages" not in st.session_state:
    st.session_state.messages = []
#建一个数组保存聊天历史

for message in st.session_state.messages:  #把聊天历史渲染在页面上
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
#st.chzt_message(messgae["role"]):创建聊天的气泡框，后续“role”=谁就代表由谁发出的聊天内容
#st.markdown():把文字渲染出来支持加粗，换行
#提前将所有对话存入到session仓库中，当页面每次刷新循环动态生成聊天气泡和对应的聊天内容
if query :=st.chat_input("请输入关于这份pdf的问题"):
#st.chat_input():页面底部的聊天输入框，用户在这里打字，回车发送
#":=":海象运算符，拿到用户的问题，赋值给query，同时判断query是否为空
    st.session_state.messages.append({"role":"user","content":query})  #<-保存用户提问
    with st.chat_message("user"):
        st.markdown(query)
#生成用户聊天气泡，同时渲染文字

    if "collection" not in st.session_state:
        with st.chat_message("assistant"):
            st.error("请先在左侧上传pdf文件哦!")
            st.session_state.messages.append({"role": "assistant", "content": "请先在左侧上传pdf文件哦!"})
#错误提示框
    else:
        with st.chat_message("assistant"):    #创建ai的聊天气泡
            results = retrieve(query,st.session_state.collection,st.session_state.bm25_model,st.session_state.chunks)
#st.session_state.collection相当于chunk_vectors文档块的向量
            answer = generate_answer(query,results)
            st.markdown(answer)

        st.session_state.messages.append({"role":"assistant","content":answer})
#将生成的答案装入session中