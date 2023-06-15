import openai
# openai.api_key = "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"
openai.api_key = "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"

# topic = "將CLI輸出寫入檔案"
topic = "生成一個以文字打印貓圖案"
myprompt = f"(英文回覆)給我一個關於'{topic}'的python程式碼，並且以英文在程式碼中註解說明，請直接輸出程式碼，不要有程式碼以外的回覆"
# myprompt = "(English response) 生成一個以文字打印貓圖案的 Python 程序"
model = "text-davinci-002"
def q1(myprompt, isPass=False):
    # print("IN:",myprompt)
    # input("Next?")
    program = myprompt
    if not(isPass):
        response = openai.Completion.create(
            engine=model,
            prompt=myprompt,
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.7,
        )

        program = response.choices[0].text.strip()
    print(program)
    with open("./run.py", "w") as f:
        f.write(program)
    input("continue?")
    a1()

def a1():
    input("prepare to run?")
    import subprocess
    result = subprocess.run(["python", "run.py"], capture_output=True)

    with open("./output.txt", "wb") as f:
        # f.write("(中文回覆)我的執行結果是否正確? 正確的話請回答'正確'，否則直接給我修正後的程式碼\n".encode('big5'))
        f.write(result.stdout)
        f.write(result.stderr)

    output = result.stdout.decode() + result.stderr.decode()
    print("output:\n",output)
    if "rror" not in output:
        # print(output)
        # print(type(output))
        print("Finish!")
        return
    q2(output)

def q2(output):
    response = openai.Completion.create(
        engine=model,
        prompt=output,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    new_code = response.choices[0].text.strip()

    print(new_code)
    q1(f"#fix my code({myprompt}): \n{new_code}", True)


q1(myprompt)