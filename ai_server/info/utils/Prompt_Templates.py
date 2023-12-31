# *_*coding:utf-8 *_*
# @Author : YueMengRui
prompt_templates = {
    "sqa": """已知信息：\n{context} \n请扮演一名专业分析师，根据上述已知信息专业的来回答用户的问题。如果无法从中得到答案，请忽略已知信息。问题是：{query}""",
    "knowledge": """已知信息：\n{context} \n请扮演一名专业分析师，根据上述已知信息专业的来回答用户的问题。不允许在答案中添加编造成分，请直接返回答案，答案请使用中文。问题是：{query}""",
    "translation": "我希望你充当语言检测器和翻译家，你将检测输入文本的语种，如果检测到是中文，就将中文翻译成英文，并返回英文。如果检测到是英文，就将英文翻译成中文，并返回中文。\n输入文本: {query}",
    "summary": "这是一个文本摘要的任务，分析并总结已知内容，然后输出摘要。\n已知内容:\n {query}",
}
