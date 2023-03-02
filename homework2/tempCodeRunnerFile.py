import random
s = [0, 7, 10, 2, 4, 9, 1, 3, 8, 6, 5, 11]
def change(s):
    # print(s)
    # print(random.randint (1, 11)) # 1~10
    switch_idx = random.randint (1, 9)
    # print(f"sw {switch_idx}")
    temp = s[switch_idx]
    temp2 = s[switch_idx+1]
    s[switch_idx+1] = s[switch_idx]
    s[switch_idx] = temp2
    # print(f"{temp} -> {temp2}",end="")
    # print(s)

    # s = [0] + s + [11]
    # print(f"{s}")
    return s
print(s)
print(change(s))