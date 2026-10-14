import os
import tempfile
import streamlit as st
from src.overlap import load_document,split_into_chunks,build_retriever,retrieve,generate_answer

st.set_page_config(page_title="我的rag智能问答助手",page_icon="📚")
st.title("📚我的rag智能问答助手")
st.write("上传一份pdf文件，我基于它的内容回答你的问题")

with st.sidebar:
    st.header("上传文档")
    uploaded_file =st.file_uploader("请选择你的pdf文件",type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        if "collection"not in st.session_state:
            with st.spinner("正在读取文档并建立语义索引..."):
                doc_text = load_document(tmp_path)
                chunks = split_into_chunks(doc_text)
                st.session_state.collection = build_retriever(chunks)
                st.success(f"知识库准备就绪，共{len(chunks)}块")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query :=st.chat_input("请输入关于这份pdf的问题"):
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(query)

    if "collection" not in st.session_state:
        with st.chat_message("assistant"):
            st.error("请先在左侧上传pdf文件哦!")
    else:
        with st.chat_message("assistant"):
            results = retrieve(query,st.session_state.collection)
            answer = generate_answer(query,results)
            st.markdown(answer)

        st.session_state.messages.append({"role":"assistant","content":answer})