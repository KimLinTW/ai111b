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
    # print(f"suffle {s} -> \n\t",end="")
    # s = [1,2,3,4,5,6,7,8,9,10]
    s = s[1:-1:1]

    random.shuffle(s)
    s = [0] + s + [11]
    # print(f"{s}")
    return s


# 計算 一個解 的分數
# s = 一個隨機解
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

def climbing(s, times):
    best_score = 0
    # 給予一個任意解
    for i in range(times):
        s2 = change(s)
        scoreS  = getScore(s)
        scoreS2 = getScore(s2)

        if scoreS2 < scoreS:
            s = s2
            print(scoreS)
            best_score = scoreS
    print(scoreS)
            

    # print(f"score {s} = {getScore(s)}")
    # print(f"score {s2} = {getScore(s2)}")

    # 調整任一解
    # 比較兩者分數

climbing(s,100000)

# def climbing(x=0, y=0, points=[(0,0)]):
#     best_x = x
#     best_y = y
#     fail = 0
#     while 1:
#         x = round(best_x,3)
#         y = round(best_y,3)
#         x += random.uniform(-0.1,0.1)
#         y += random.uniform(-0.1,0.1)

#         if getDis(x,y, points) < getDis(best_x, best_y, points):
#             best_x = x
#             best_y = y
#             fail = 0
#         else: fail+=1

#         if fail >= 100: break
#     print(f"Total Distance:{getDis(best_x, best_y, points, 1)}")
#     return [best_x,best_y]
    
    
