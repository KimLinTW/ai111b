# 此程式100%原創 未參考chatGPT、其他網站
import matplotlib.pyplot as plt
import numpy as np
import random
import math

points = [(0,1.9),(1,3.1),(2,3.9),(3,5.0),(4,6.2)]

def climbing(x=0, y=0, points=[(0,0)]):
    best_x = x
    best_y = y
    fail = 0
    while 1:
        x = round(best_x,3)
        y = round(best_y,3)
        x += random.uniform(-0.1,0.1)
        y += random.uniform(-0.1,0.1)

        if getDis(x,y, points) < getDis(best_x, best_y, points):
            best_x = x
            best_y = y
            fail = 0
        else: fail+=1

        if fail >= 100: break
    print(f"Total Distance:{getDis(best_x, best_y, points, 1)}")
    return [best_x,best_y]

def getDis(a,b, points, times=2):  # a,b 是 y=ax+b 的a、b
    dis_record = []
    if times!=2:print(f"y = {a}x + {b}")
    total = 0
    for point in points:
        x = point[0]
        y = point[1]
        a_ = a
        b_ = -1
        c_ = b

        dis = abs(a_*x+b_*y+c_) / (a_**2+b_**2)**0.5
        dis_record.append(dis)
        if times != 2 :
            # print(f"point:{point} : {abs(a_*x+b_*y+c_)} / { math.sqrt(a_**2+b_**2)}  = {dis}", end="\n")
            # print("dis",*dis_record)
            pass
        total += abs(dis)**times
    return total

a, b = climbing(0,0,points)

x = np.linspace(0, 10, 100)
y = a * x + b
points = [(0,1.9),(1,3.1),(2,3.9),(3,5.0),(4,6.2)]
xs, ys = zip(*points)

plt.plot(x, y, color='red')
plt.scatter(xs, ys)
plt.xlim(-0.5, 5)
plt.ylim(0, 9)

plt.title('y = {}x + {}'.format(round(a,2), round(b,2)))
plt.xlabel('x')
plt.ylabel('y')

plt.show()
