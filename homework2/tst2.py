# 此程式100%原創 未參考chatGPT、其他網站
import random

s = [1,2,3,4,5,6,7,8,9,10]
random.shuffle(s)
s = [0] + s + [11]
# print(s)

points = [
    (0,0),
    (1,0),
    (2,0),
    (3,0),
    (0,1),
    (1,1),
    (2,1),
    (3,1),
    (0,2),
    (1,2),
    (2,2),
    (3,2)
]

def getDis(p1, p2):
    dis = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
    # print(f"p1{p1}, p2{p2}, dis:{dis}")
    return dis

def change(s):
    # print(random.randint (1, 11)) # 1~10
    switch_idx = random.randint (1, 9)
    temp = s[switch_idx]
    temp2 = s[switch_idx+1]
    s[switch_idx+1] = s[switch_idx]
    s[switch_idx] = temp2
    return s

def getScore(s):
    score = 0
    for i in s:
        try: s[i+1]
        except: break
        node1 = points[s[i]]
        node2 = points[s[i+1]]
        # print(score,end=" ")
        score += getDis(node1,node2)
    # print(score)
    return score

def climbing(s, times, exit=2000):
    timer = 0
    best_score = 0
    # 給予一個任意解
    for i in range(times):
        s2 = change(s.copy())
        scoreS  = getScore(s)
        scoreS2 = getScore(s2)

        if scoreS2 < scoreS:
            s = s2
            print(scoreS)
            best_score = scoreS
        else:
            timer += 1  
            if timer >= exit: break
    print(scoreS)

climbing(s,100000)
