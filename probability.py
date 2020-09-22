#coding=utf-8

"""
Algorithms about probability
Author: 段凯强
"""

import math

def entropyOfList(ls):
    """
    Given a list of some items, compute entropy of the list
    The entropy is sum of -p[i]*log(p[i]) for every unique element i in the list, and p[i] is its frequency
    """
    elements = {}
    for e in ls:
        elements[e] = elements.get(e, 0) + 1
        #print(elements[e])

    length = float(len(ls))
    return sum([-v/length*math.log(v/length) for v in list(elements.values())])

'''
l = ['到', '有', '过', '的', '了']
r = ['人', '人', '人', '人', '人']

print('left', entropyOfList(l))
print('right', entropyOfList(r))'''


