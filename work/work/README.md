\# 小问答助手：ReAct 智能体 + 混合检索 PDF 问答



一个基于 \*\*ReAct 范式\*\* 的智能问答助手。既能做基础聊天（算数、看时间），也能上传 PDF 后基于文档内容回答问题。



界面用 Streamlit，模型用 DeepSeek，检索用 \*\*向量检索 + BM25 关键词检索\*\* 的混合方案。



\## 功能



\- 💬 \*\*基础聊天\*\*：调用工具完成算数、查询当前时间

\- 📄 \*\*PDF 问答\*\*：上传 PDF，自动清洗、切块、建索引，基于内容回答

\- 🔁 \*\*ReAct 循环\*\*：模型自主决定"思考 → 调工具 → 看结果 → 再思考"，最多 5 步

\- 🔍 \*\*混合检索\*\*：向量检索（语义）+ BM25（关键词），用 RRF 融合排序



\## 截图



!\[界面截图](screenshot1.png)



!\[问答截图](screenshot2.png)



> 截图放在项目根目录，文件名分别是 `screenshot1.png`、`screenshot2.png`。



\## 技术方案



| 模块 | 选型 | 理由 |

|---|---|---|

| 界面 | Streamlit | 纯 Python 写前端，适合快速做 demo |

| 大模型 | DeepSeek-Chat | 中文能力强，价格便宜，接口兼容 OpenAI |

| 向量模型 | BAAI/bge-small-zh-v1.5 | 中文语义检索效果好，模型小、本地可跑 |

| 向量库 | ChromaDB | 轻量、无需部署，适合本地原型 |

| 关键词检索 | jieba + rank\_bm25 | 弥补向量检索对专有名词、数字不敏感的问题 |

| 融合策略 | RRF（倒数排名融合） | 不依赖分数尺度，简单稳定 |

| PDF 解析 | PyMuPDF | 提取速度快，能拿到逐页文本 |



\## 数据流



```text

用户提问

&#x20;  │

&#x20;  ▼

run\_agent()  ← ReAct 循环（最多 5 步）

&#x20;  │  模型输出 Thought / Action / Action Input

&#x20;  ├── calculator ──► 安全校验后 eval

&#x20;  ├── get\_time  ──► 返回当前时间

&#x20;  └── search\_pdf

&#x20;         │

&#x20;         ▼

&#x20;  PDF 加载（PyMuPDF）

&#x20;         │

&#x20;         ▼

&#x20;  LLM 清洗（修复排版、表格转 Markdown）

&#x20;         │

&#x20;         ▼

&#x20;  文本切块（500 字 / 重叠 100）

&#x20;         │

&#x20;         ├──► 向量化 → ChromaDB

&#x20;         └──► jieba 分词 → BM25

&#x20;         │

&#x20;         ▼

&#x20;  混合检索（各取 top10 → RRF 融合 → 取 top5）

&#x20;         │

&#x20;         ▼

&#x20;  拼接上下文 → LLM 生成答案（只提取、不计算）

&#x20;         │

&#x20;         ▼

&#x20;  Final Answer 返回给用户

```



\## 项目结构



```text

work/

├── main.py              # Streamlit 入口页

├── 1\_basic\_run.py       # 基础聊天页面

├── 2\_pdf\_search.py      # PDF 检索页面

├── agent.py             # ReAct 循环 + 工具注册

├── search.py            # PDF 解析、切块、混合检索、答案生成

├── tools.py             # calculator / get\_time

├── requirements.txt

├── .gitignore

└── README.md

```



\## 快速开始



```bash

\# 1. 安装依赖

pip install -r requirements.txt



\# 2. 配置 API Key

\# 在项目根目录新建 .env 文件，写入：

\# DEEPSEEK\_API\_KEY=你的key



\# 3. 启动

streamlit run main.py

```



首次使用 PDF 检索时，会加载向量模型并清洗文档，需要等待一会儿。



\## 已知不足



\- \*\*检索质量依赖清洗\*\*：PDF 表格错位严重，靠 LLM 清洗，成本和效果不稳定

\- \*\*切块按字符硬切\*\*：可能把一句话、一个表格切断，影响检索命中

\- \*\*ChromaDB 存在内存里\*\*：重启后索引丢失，每次都要重建

\- \*\*无重排（rerank）\*\*：融合后直接取 top5，没有精排环节

\- \*\*对话无记忆\*\*：每次 `run\_agent` 都从新的 messages 开始，多轮对话不连贯

\- \*\*`calculator` 用 eval\*\*：虽然做了字符白名单校验，仍不算最安全的方案

\- \*\*错误处理较粗\*\*：检索失败只返回一句"系统维护中"，不利于排查



\## Roadmap



\- \[ ] 用语义切块（按段落/标题）替代固定字符切块

\- \[ ] 接入 rerank 模型（如 bge-reranker）做精排

\- \[ ] 把 ChromaDB 持久化到本地目录，避免重复建索引

\- \[ ] 给 `run\_agent` 加对话历史，支持多轮

\- \[ ] 用 `ast.literal\_eval` 或专用表达式解析替换 `eval`

\- \[ ] 给 PDF 清洗加缓存，减少重复调用

