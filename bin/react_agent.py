import re
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 1. 定义工具 ==========
def calculator(expr: str) -> str:
    try:
        # 简单过滤一下，只允许数字和基本运算
        if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
            return "错误：表达式含非法字符"
        return str(eval(expr))
    except Exception as e:
        return f"计算错误：{e}"

def get_time(_: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOLS = {
    "calculator": calculator,
    "get_time": get_time,
}

# ========== 2. 系统提示词 ==========
SYSTEM_PROMPT = """你是一个 ReAct 智能体。可用工具：
1. calculator：计算数学表达式，输入如 "2+3*4"
2. get_time：获取当前时间，输入留空

严格按以下格式输出：
Thought: 你的思考
Action: 工具名
Action Input: 工具输入

当能给出最终答案时，输出：
Final Answer: 最终答案"""

# ========== 3. 调用大模型 ==========
def call_llm(messages):
    # 这里把 stream 设为 False，方便直接拿完整文本做正则解析
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content

# ========== 4. 主循环（ReAct核心） ==========
def run_agent(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    for step in range(5): # 最多循环5次，防止死循环
        print(f"\n--- 第 {step+1} 步 ---")
        output = call_llm(messages)
        print(output)
        messages.append({"role": "assistant", "content": output})

        # 判断是否给出最终答案
        if "Final Answer:" in output:
            print("\n✅ 最终答案：", output.split("Final Answer:", 1)[1].strip())
            return

        # 判断是否要调用工具
        action_match = re.search(r"Action:\s*(\w+)", output)
        input_match = re.search(r"Action Input:\s*(.+)", output)

        if action_match and input_match:
            tool_name = action_match.group(1).strip()
            tool_input = input_match.group(1).strip()
            
            print(f"-> 调用工具：{tool_name}，参数：{tool_input}")
            
            if tool_name in TOOLS:
                obs = TOOLS[tool_name](tool_input)
            else:
                obs = f"错误：未知工具 {tool_name}"
            
            print(f"-> 观察结果：{obs}")
            # 把工具执行的结果塞回对话，让大模型继续思考
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            messages.append({"role": "user", "content": "请严格按格式输出，或给出 Final Answer。"})

if __name__ == "__main__":
    run_agent("现在几点了？然后帮我算一下 (123 + 456) * 789 / 3")