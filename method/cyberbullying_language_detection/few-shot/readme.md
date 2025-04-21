Firstly install OpenPrompt https://github.com/thunlp/OpenPrompt
Then copy prompts/knowledgeable_verbalizer.py to Openprompt/openprompt/prompts/knowledgeable_verbalizer.py

example shell scripts:

python fewshot.py --result_file ./output_fewshot.txt --dataset clickbait --template_id 0 --seed 123 --shot 10 --verbalizer manual

python fewshot1.py --result_file ./output_fewshot1.txt --dataset snippets --template_id 0 --seed 144 --verbalizer cpt --calibration

Note that the file paths should be changed according to the running environment. 

The datasets are downloadable via OpenPrompt.
