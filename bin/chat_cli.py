import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def chat_loop():
    print("--- AI 助手已启动 (输入 quit 退出) ---")
    messages = [{"role": "system", "content": "你是一个乐于助人的AI助手。"}]
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        messages.append({"role": "user", "content": user_input})
        
        print("AI: ", end="", flush=True)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )
        
        full_reply = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full_reply += text
        print()
        messages.append({"role": "assistant", "content": full_reply})

if __name__ == "__main__":
    chat_loop()