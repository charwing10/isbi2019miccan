#!/usr/bin/env python
"""Different Network Structures"""

__author__ = "Qiaoying Huang"
__date__ = "04/08/2019"
__institute__ = "Rutgers University"


import scipy.stats as ss
import numpy as np


def gaussiansample(l, std, size):
    mean = int(l/2)
    x = np.arange(-mean, mean)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale = std) - ss.norm.cdf(xL, scale = std)
    prob = prob / prob.sum() #normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size = size, p = prob) + mean
    nums = np.unique(nums)
    nums = nums.tolist()
    while len(nums)<size:
        new = np.random.choice(x, size = 1, p = prob) + mean
        if new[0] not in nums:
            nums.append(new[0])
    nums = np.asarray(nums)
    eightlowest = np.asarray([mean-3, mean-2, mean-1, mean, mean+1, mean+2, mean+3, mean+4], dtype=np.int)
    for i in range(len(eightlowest)):
        index = np.argwhere(nums == eightlowest[i])
        nums = np.delete(nums, index)
    np.random.shuffle(nums)
    sampleidx = nums[0:(size-8)]
    sampleidx = np.append(sampleidx, eightlowest)
    return sampleidx