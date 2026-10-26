from tools import *
from search import *

rag_cache = {}

def call_llm(messages):
    resp = client.chat.completions.create(
        model = "deepseek-chat",
        messages = messages,
        temperature = 0
    )
    return resp.choices[0].message.content

def run_agent(question):
    messages=[{"role":"system","content":SYSTEM_PROMPT},
              {"role":"user","content":question},]
    for step in range(5):
        print(f"第{step+1}步...")
        output = call_llm(messages)
        print(output)
        messages.append({"role":"assistant","content":output})
        action_match = re.search(r"Action:\s*(\w+)",output)
        input_action = re.search(r"Action Input:\s*(.+)",output)
        if action_match and input_action:
            tool_name = action_match.group(1).strip()
            tool_input = input_action.group(1).strip()
            print(f"调用工具{tool_name},参数{tool_input}")
            if tool_name in TOOLS:
                obs = TOOLS[tool_name](tool_input)
            else:
                obs = f"错误，未知工具{tool_name}"
            print(f"观查结果：{obs[:100]}")
            messages .append( {"role":"user","content":obs})
            continue
        if "Final Answer:" in output:
            final_ans = output.split("Final Answer:")[-1].strip()
            print("\n最终结果：",final_ans)
            return final_ans

        if not action_match:
            print("\n最终结果：",output)
            return output

    time_msg = "超过最大迭代，请重新组织语言再来问我吧"
    print(time_msg)
    return time_msg

def search_pdf(query_str):
    global rag_cache
    try:
        if not rag_cache:
            print("(首次加载：正在读取并清洗pdf请稍后...)")
            doc_text = load_document(PDF_PATH)
            chunks = split_into_chunks(doc_text)
            collection,bm25_model,chunks =build_retriever(chunks)

            rag_cache['collection'] = collection
            rag_cache['bm25_model'] = bm25_model
            rag_cache['chunks'] = chunks

        results = retrieve(query_str,rag_cache['collection'],rag_cache['bm25_model'],rag_cache['chunks'])
        return generate_answer(query_str,results)
    except Exception as e:
        return f"【系统级错误】：检索工具发生底层代码异常{e}，请不要重试，直接告述用户系统维护中"

TOOLS = {
    "calculator":calculator,
    "get_time":get_time,
    "search_pdf":search_pdf
}
SYSTEM_PROMPT = """你是一个 ReAct 智能体。可用工具：
1. calculator：计算数学表达式，输入如 "2+3*4"
2. get_time：获取当前时间，输入留空
3. search_pdf:查询文档(如查学分、查课程、查专业课代码等)

严格按以下格式输出：
Thought: 你的思考
Action: 工具名
Action Input: 工具输入

当能给出最终答案时，输出：
Final Answer: 最终答案"""