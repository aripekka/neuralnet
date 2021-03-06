from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from network import Network
import random

'''
A test to train a simple network to recognise whether a point is inside
a circle or not.
'''

nw = Network((2,4,1))

trainingset = []
for i in xrange(1000):
    x = random.uniform(-3,3)
    y = random.uniform(-3,3)

    point = ([x,y],[0])

    if np.sqrt(x**2+y**2) < 1:
        point[1][0] = 1
    trainingset.append(point)
    
nw.train(trainingset,50,2,1)

#print nw

x = np.linspace(-5,5,100)
         
y = np.zeros(x.shape)
y2 = np.zeros(x.shape)
y3 = np.zeros(x.shape)    
for i in xrange(x.size):
    outp = nw.compute_outputs((x[i],0))
    y[i] = outp[-1][0]
    outp = nw.compute_outputs((0,x[i]))
    y2[i] = outp[-1][0]
    outp = nw.compute_outputs((x[i]/np.sqrt(2),x[i]/np.sqrt(2)))
    y3[i] = outp[-1][0]

plt.plot(x,y,label='y = 0')
plt.plot(x,y2,label='x = 0')
plt.plot(x,y3,label='diagonal')
plt.legend()
plt.show()    
    
