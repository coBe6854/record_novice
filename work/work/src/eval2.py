from src.search_test2 import*
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
doc_path = os.path.join(BASE_DIR,"data","2025级计算机科学基础学科拔尖班专业培养方案(主修).pdf")
eval_data_path = os.path.join(BASE_DIR,"data","eval_dataset.txt")
#加载评测数据函数，读取测试题
def load_eval_dataset(path):
    dataset = []
#定义一个空的数据集
    if not os.path.exists(path):
        print(f"找不到测试集文件：{path}")
        return dataset
    with open(path, 'r', encoding="utf-8") as f:
#with：能自动关闭文件不用再手动关闭，释放资源
#open(path,'r'):只读模式打开
#encoding = "utf-8":用utf-8编码读取，防止中文乱码
#as f：取别名为f
        for line in f:
            line = line.strip()
#.strip():删掉这行的开头和结尾的空格和换行符
            if not line or '|' not in line:
#not line：这一行是空格
#‘|’not in line：这一行没有|
                continue
            question,answer = line.split('|',1)
#split('|',1):只在第一个‘|’分割，防止标准答案里有|导致切错
            dataset.append({"question":question.strip(),"answer":answer.strip()})
    return dataset

def evaluate():
    if not os.path.exists(doc_path):
        print(f"找不到PDF文件：{doc_path}")
        return
    print("正在加载资源库...")
    doc_text = load_document(doc_path)
#读取doc_path = "data/2025级计算机科学基础学科拔尖班专业培养方案(主修).pdf"
    chunks = split_into_chunks(clean_with_llm(doc_text))
    collection,bm25_model,chunks = build_retriever(chunks)
    print(f"知识库准备就绪，共{len(chunks)}块。")

    test_dataset = load_eval_dataset("data/eval_dataset.txt")#<-此处已将dataset这个装着字典的列表的数据传给了test_dataset
#导入了eval.text里的问题及判断标准
#加载测试题目
#test_dataset:是个列表，但他后续装的是字典
    if not test_dataset:
        print("测试集为空，请检查eval.txt内容。")
        return

    correct_count = 0
    total = len(test_dataset)

    print(f"开始考试!总共{total}道题：\n")
    for item in test_dataset:
        query = item["question"]
        expected_answer = item["answer"]
        results = retrieve(query, collection, bm25_model, chunks, top_k=3)
        answer = generate_answer(query,results)
        is_correct = expected_answer in answer
        if is_correct:
            correct_count += 1
            print(f"命中|问：{query}|答：{answer[:50]}...")
        else:
            print(f"未命中|问：{query}|答：{expected_answer}|ai答：{answer[:50]}...")
#只打印回答前的50个字符，控制台不会刷超长文本

    accuracy = correct_count / total
    print(f"\n评测结束！总题数：{total}，命中数：{correct_count}")
    print(f"准确率：{accuracy:.2%}")
