import os
import openai

openai.api_key = "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"

# topic = "將CLI輸出寫入檔案"
# myprompt = f"給我一個關於'{topic}'的python程式碼，並且以繁體中文在程式碼中註解說明，不要有程式碼以外的回覆"

myprompt = "請告訴我如何透過 openai套件 要求chatGPT產生hello python程式碼、預期結果,並在本地端將CLI回傳的程式碼保存到./run.py執行,並把執行結果寫入output.txt,包括報錯時的輸出,再把執行結果傳回chatGPT,檢查是否需要修正bug"

response = openai.Completion.create(model="text-davinci-003", prompt=myprompt, temperature=0, max_tokens=300)
print(response['choices'][0]['text'])

"""
透過chatGPT產生python程式碼、預期結果，並在本地端將CLI回傳的程式碼保存到./run.py執行，並把執行結果寫入output.txt，包括報錯時的輸出，再把執行結果傳回chatGPT，檢查是否需要修正bug




1. 透過
pre = f"給我一個關於{topic}的python程式碼，並且以繁體中文在程式碼中註解說明，不要有程式碼以外的回覆"

"""