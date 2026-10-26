import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url = "https://api.deepseek.com")

def calculator (expr:str)->str:
    try:
        if not re.fullmatch(r"[0-9+\-*/\s().]+",expr):
            return "错误非法字符，请重试..."
        return str(eval(expr))
    except Exception as e:
        return f"计算错误：{e}"

def get_time(_:str)->str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


