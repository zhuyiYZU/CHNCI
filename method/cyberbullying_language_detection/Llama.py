import torch
from modelscope import snapshot_download
from modelscope import AutoModelForCausalLM
from transformers import AutoTokenizer
import csv
import transformers

model_dir = snapshot_download('FlagAlpha/Llama3-Chinese-8B-Instruct')
pipeline = transformers.pipeline(
    "text-generation",
    model=model_dir,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",

)





reader_file = r'总数据.csv'
writer_file = 'lama_output.csv'



with open(reader_file,mode='r',newline='',encoding='gbk') as infile,\
    open(writer_file,mode='w',newline='',encoding='utf-8') as outfile:
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)
    next(csv_reader)

    for row in csv_reader:
        messages = []

        classification_prompt = (f"我现在要进行隐性恶性语言检测，帮我根据隐性恶性语言的不明显的特点对输入的网络评论进行解释并帮我判断输入的文本时友善的还是恶意的。\n"
                                 f"如果是恶意的就输出1,如果是友善的就输出0。\n")

        classification_prompt += f"输入文本:{row[3]}"
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
        row.append(response_LLM)
        csv_writer.writerow(row)





