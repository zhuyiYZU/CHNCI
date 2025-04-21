import csv
import openai
import re
from openai import OpenAI

import torch
from modelscope import snapshot_download
from modelscope import AutoModelForCausalLM
from transformers import AutoTokenizer
import csv
import transformers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 初始化模型（保持不变）
model_dir = snapshot_download('FlagAlpha/Llama3-Chinese-8B-Instruct')
pipeline = transformers.pipeline(
    "text-generation",
    model=model_dir,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)

# key = "sk-riNBapvBKim4P26h71GZY9vwgRQIvIN8aZvFkrG2DGIHISNK"
# #
# openai.api_key = key
# # openai.base_url = 'https://api.deepseek.com'
# openai.base_url = 'https://chatapi.littlewheat.com/v1'


original_file = 'dataset/sampled_output.csv'
different_file = 'result/sampled_output_m-agent.csv'


# prompts = ["对于网络上出现的热点事件，网友会对这一事件自由的发表自己的观点，但是网友发表的观点评论可能会有直接或间接冒犯到个人或集体的，帮我对以下输入的网友评论进行解析，并判断是友善的还是恶意的。",
#           "现在社交媒体对带有明显恶意、辱骂等负面词语的评论的屏蔽已经有了较为完善的系统，但是对于一些在表面上看似无害、模糊不清，但实际带有负面或恶意含义的语言，往往隐藏在正常的表达中，使得它们更难被直接识别。帮我对以下输入的评论进行解析并帮我识别出是友善的还是恶意的。",
#           "请分别从情感倾向、主题和话题、目标对象、潜在意图、上下文关联、话语结构、标点符号和用户背景等方面对以下输入的网络评论进行全面分析解释，并判断该语句是友善的还是恶意的，给出标签。",
#           "你的任务是检测以下输入的微博评论中的恶性语言。恶性语言可能会有通过讽刺、暗示等微妙的方式表现，并不直接使用侮辱性词汇。我需要你结合上下文、用户互动等方面，对评论进行细致的分析解释，并识别这些是否恶性语言。",
#           "当针对某一事件的大量网络评论突然涌现会造成网络暴力，对网络环境的纯净带来了巨大挑战。帮我根据输入的评论进行联想到相应的事件，对评论进行公正的解析并判断该评论是友善的还是恶意的。"]

# prompts = ["For trending events on the internet, netizens freely express their opinions about these events. However, their comments might directly or indirectly offend individuals or groups. Please help analyze the following netizens' comments and determine whether they are friendly or malicious. ",
#           "Social media now has relatively robust systems for filtering comments with obvious malicious or abusive words. However, language that appears harmless or ambiguous on the surface but actually carries negative or malicious intent often hides within normal expressions, making it harder to identify directly. Please help analyze the following comments and identify whether they are friendly or malicious.",
#           "Please conduct a comprehensive analysis and interpretation of the following online comments from the perspectives of sentiment tendency, theme and topic, target, potential intent, contextual relevance, discourse structure and punctuation, as well as user background. Then determine whether the comment is friendly or malicious and provide a label. ",
#           "Your task is to detect malicious language in the following Weibo comments. Malicious language may be expressed subtly through sarcasm, insinuation, or other indirect means without directly using offensive words. I need you to conduct a detailed analysis and interpretation of the comments, considering the context, user interactions, and other aspects, to identify whether they contain malicious language.",
#           "When a large number of online comments about a particular event suddenly emerge, it can lead to cyberbullying, posing a significant challenge to maintaining a clean online environment. Please help associate the input comments with the relevant event, conduct an impartial analysis of the comments, and determine whether they are friendly or malicious."]
prompts = ["请阅读以下评论并判断其是否包含任何形式的网络欺凌。提供你的理由并输出最终标签：""1表示网络欺凌""或""0表示非网络欺凌""。",
          "分析以下文本并判断其是否表现出网络欺凌行为。解释你的理由并给出标签（""1""表示网络欺凌，""0""表示非网络欺凌）。",
          "你是在线安全专家。请审阅该评论并判断是否应将其归类为网络欺凌。论证你的答案并输出分类结果。（""1""表示网络欺凌，""0""表示非网络欺凌）",
          "评估给定评论并判断其是否构成网络欺凌。包含你的推理过程并以标签作结：网络欺凌或非网络欺凌。（""1""表示网络欺凌，""0""表示非网络欺凌）",
          "判断以下评论是否为网络欺凌的实例。写出简要解释并分配标签（网络欺凌/非网络欺凌）。（""1""表示网络欺凌，""0""表示非网络欺凌）"]

start_row = 9592
current_row = 0
with open(original_file, mode='r', newline='', encoding='utf-8') as infile, \
        open(different_file, mode='a', newline='', encoding='utf-8') as outfile:
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)

    for row in csv_reader:
        current_row+=1
        if current_row<start_row:
            continue

        if ("0" in str(row[2])) or ("1" in str(row[2])):
            write_row = [row[2], row[3]]
            comment = str(row[3])  # 原评论文本

            for prompt in prompts:
                inter_analysis = {}
                messages = []
                for i in range(1, 4):
                    classification_prompt = prompt
                    classification_prompt += f"\n输入评论：{row[3]}\n"
                    classification_prompt += f"注意:输出应该是json格式。\n"
                    classification_prompt += f"如：\n {{ \n\"解释\":[],\n\"标签\":[（判断非网络欺凌输出“0”，判断网络欺凌输出“1”）]\n}}\n"
                    classification_prompt += f"注意:回答的解释和标签用中括号“[]”扩出来，解释和标签均不可以为空。必须给出一组解释和标签！\n"

                    text_with_prompt = classification_prompt
                    messages.append({"role": "user", "content": text_with_prompt})
                    prompt = pipeline.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    terminators = [
                        pipeline.tokenizer.eos_token_id,
                        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = pipeline(
                        prompt,
                        max_new_tokens=512,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9
                    )

                    response_LLM = outputs[0]["generated_text"][len(prompt):]


                    print('LLM:')
                    print(response_LLM)

                    if response_LLM:
                        LLM_analysis_label = re.findall(r'\".*?\":([^,}]*)', response_LLM)
                        LLM_analysis = "NULL"
                        LLM_label = "NULL"
                        LLM_analysis = LLM_analysis_label[0].strip() if len(LLM_analysis_label) > 0 else "NULL"
                        LLM_label = LLM_analysis_label[1].strip() if len(LLM_analysis_label) > 1 else "NULL"
                        if '1' in LLM_label:
                            LLM_label = '1'
                        elif '0' in LLM_label:
                            LLM_label = '0'
                        else:
                            LLM_label = 'NAN'
                    else:
                        LLM_analysis = "NAN"
                        LLM_label = "NAN"

                    write_row.append(LLM_label)
                    write_row.append(LLM_analysis)
                    print("=" * 100)
                    print(write_row)

            csv_writer.writerow(write_row)
            outfile.flush()





