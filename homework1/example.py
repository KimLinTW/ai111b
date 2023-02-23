import matplotlib.pyplot as plt
import numpy as np
import random

x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

def predict(a, xt):
	return a[0]+a[1]*xt

def MSE(a, x, y):
	total = 0
	for i in range(len(x)):
		total += (y[i]-predict(a,x[i]))**2
	return total

def loss(p):
	return MSE(p, x, y)

def optimize():
    x1 = 0.0
    y1 = 0.0
    bx = x1
    by = y1
    fail = 0
    while 1:
        x1 = bx
        y1 = by
        x1 += random.uniform(-0.1,0.1)
        y1 += random.uniform(-0.1,0.1)
        if loss([x1, y1]) < loss([bx, by]):
            bx = x1
            by = y1
            fail = 0
        else :
            fail += 1
        if fail >= 10: break
    p = [bx, by]
    print(p)
    
    return p

p = optimize()

# Plot the graph
y_predicted = list(map(lambda t: p[0]+p[1]*t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
