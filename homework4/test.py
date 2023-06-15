import subprocess
import chardet

# 定義要執行的指令
command = 'dir'

# 執行指令，並將輸出結果存入變數result
result = subprocess.check_output(command, shell=True)

# 判斷編碼方式
encoding = chardet.detect(result)['encoding']

# 將輸出結果寫入檔案
with open('output.txt', 'w', encoding=encoding) as f:
    f.write(result.decode(encoding))
