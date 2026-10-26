import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import streamlit as st
from agent import run_agent  # 导入你的 Agent 大脑

st.set_page_config(page_title="智能助理", page_icon="💬")
st.title("💬 基础智能聊天助手")
st.write("我可以陪你聊天，也可以帮你算数和看时间！")

# 1. 初始化聊天记忆
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 渲染历史记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 接收输入并调用 Agent
if query := st.chat_input("试着问我：现在几点？或者 123*456 等于多少？"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 这里直接调用你的 agent 跑工具循环
            ans = run_agent(query)
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})