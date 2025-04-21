import torch
from modelscope import snapshot_download
from modelscope import AutoModelForCausalLM
from transformers import AutoTokenizer
import csv
import transformers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

model_dir = snapshot_download('FlagAlpha/Llama3-Chinese-8B-Instruct')
pipeline = transformers.pipeline(
    "text-generation",
    model=model_dir,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)

reader_file = r'sampled_output.csv'
writer_file = 'sampled_lama_output_para_with_explanation.csv'

true_labels = []
predicted_labels = []

with open(reader_file, mode='r', newline='', encoding='utf-8') as infile, \
        open(writer_file, mode='w', newline='', encoding='utf-8') as outfile:
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)
    header = next(csv_reader)
    header.extend(["Predicted Label", "Model Explanation"])  # Add columns for prediction and explanation
    csv_writer.writerow(header)

    for row in csv_reader:
        messages = []

        # Assuming the true label is in column 2 (adjust as needed)
        true_label = int(row[2])  # Change index to match your true label column

        classification_prompt = (
            "请对以下文本进行改写，使其语义更加清晰，然后根据改写后的内容解释文本含义，并判断是否为网络暴力。"
            "请先给出详细的分析解释，然后在最后一行单独用'结论：'开头，后面跟上0或1的判定结果（1表示构成网络暴力，0表示不构成）。"
        )
        classification_prompt += f"\n\n输入文本:\n{row[3]}"  # Assuming text is in column 3
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
        print(response_LLM)
        # Extract prediction and explanation from response
        prediction = 0
        explanation = response_LLM

        # Try to find the conclusion in the response
        conclusion_match = re.search(r'结论：\s*([01])', response_LLM)
        if conclusion_match:
            prediction = int(conclusion_match.group(1))
            # Remove the conclusion line from explanation for cleaner output
            explanation = re.sub(r'结论：\s*[01]', '', explanation).strip()
        else:
            # Fallback: look for 0 or 1 in the response if conclusion format not found
            if "1" in response_LLM:
                prediction = 1
            elif "0" in response_LLM:
                prediction = 0

        true_labels.append(true_label)
        predicted_labels.append(prediction)

        # Add both prediction and explanation to the row
        row.extend([prediction, explanation])
        csv_writer.writerow(row)

# Calculate metrics
if len(set(true_labels)) > 1:  # Check if we have both classes
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
else:
    print("Warning: Only one class present in true labels. Cannot compute metrics.")