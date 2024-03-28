from transformers import AutoTokenizer, AutoModel
from utils import load_model_on_gpus
import json
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = load_model_on_gpus("THUDM/chatglm-6b", num_gpus=2)
model = model.eval()

# read csv
df = pd.read_csv("data/bundle_intent.csv")
#pd to list
intent = df['intent'].tolist()

response, history = model.chat(tokenizer, "I will provide a description of the clothing set I need. please break it down into 3-5 items.[Format: item1 | item2 | item3 | item4]", history=[])
print(response)
response, history = model.chat(tokenizer, "Dress up for a Party", history=history)
print(response)
history[1] = (history[1][0], 'Clothesline | Dress | Accessories | Pumps')
response, history = model.chat(tokenizer, "something for a little girl", history=history)
history[2] = (history[2][0], 'Dress | Hat | Dress | shoes | hair accessories')
print("history: ")
print(history)


# 判断一句话是否有中文
def is_contain_chinese(check_str):
    """
    判断一句话是否有中文
    :param check_str: {str} 需要检测的字符串
    :return: {bool} 是否有中文
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

idx = 0
res = []
for i in intent:
    print("intention: " + i)
    n = 0
    while True:
        n += 1
        if n == 5:
            response = []
            break
        
        response, _ = model.chat(tokenizer, i, history=history)
        print(response)
        if is_contain_chinese(response):
            continue
        
        # 将文字用'|'分割，忽略前后空格
        response = response.split('|')
        response = [i.strip() for i in response]
        print(response)
        break
    res.append({'bundle ID': idx,
                'intent': i ,
                'tokens': response})
    idx += 1
    
# save to json
with open('data/bundle_intent_token.json', 'w') as f:
    json.dump(res, f)