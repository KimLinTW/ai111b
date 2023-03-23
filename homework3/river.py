# 此程式100%原創 未參考chatGPT、其他網站
from copy import *
from random import *
"""
2. 過河問題

    start
        人 狼 羊 菜
        0  0  0  0

    finish
        人 狼 羊 菜
        1  1  1  1

    避免:
        只有 狼 羊
        [x 1 1 x] and 1的數量==2
        [x 0 0 x] and 0的數量==2
        只有 羊 菜
        [x x 1 1] and 1的數量==2
        [x x 0 0] and 0的數量==2
"""

state = [0,0,0,0]
finall = [1,1,1,1]

def swap(i):
    if i:return 0
    return 1

def get_valid_act(filter, arr, human=0):
    output = []
    for i in range(len(arr)-1):
        if arr[i+1] == filter: output.append(i+1)
    if human != 0 and arr[0]==filter: output.append(0)
    return output

def valid_next_state(state):
    act_set = []
    # if 人 == 0
    if state[0] == 0:
        valid_act = get_valid_act(state[0], state)
        for i in valid_act:
            s = deepcopy(state)
            s[i] = swap(s[i])
            s[0] = swap(s[0])
            act_set.append(s)

    # if 人 == 1
    if state[0] == 1:
        valid_act = get_valid_act(state[0], state)
        for i in valid_act:
            s = deepcopy(state)
            s[i] = swap(s[i])
            s[0] = swap(s[0])
            act_set.append(s)

    state[0] = swap(state[0])
    act_set.append(state) 

    # remove invalid action
    for i in act_set:
        if i[1] == i[2] and len(get_valid_act(i[1], i, 1))==2:
            act_set.remove(i)
            
        elif i[2] == i[3] and len(get_valid_act(i[2], i, 1))==2:
            act_set.remove(i)
    return list(map(list, set(map(tuple, act_set))))

# print(valid_next_state([0,0,0,0]))
# print(valid_next_state([1,0,1,0]))
# print(valid_next_state([0,0,1,0]))
# print(valid_next_state([1,1,1,0]))
# print(valid_next_state([0,1,0,0]))
# print(valid_next_state([1,1,0,1]))
# print(valid_next_state([0,1,0,1]))
# print(valid_next_state([1,1,1,1]))

path = []
def bfs(state, path):
    if f"{state} -> " in path: return  # 環路預防
    path.append(f"{state} -> ")
    if state == [1,1,1,1]: 
        print(f"find:{f'{path}'[0:-5]}")
        return path
    nxt_state = valid_next_state(state)
    for i in nxt_state: 
        bfs(i, deepcopy(path))
    return f"{state}"

state = [0,0,0,0] 
bfs(state, path)