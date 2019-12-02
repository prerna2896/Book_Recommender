import csv
import numpy as np 
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ratings = np.loadtxt("../data/ratings.csv", delimiter=',')

#Printing stats for the ratings
print(stats.describe(ratings[:,2]))

#Printing distribution of ratings
val = ratings[:,2].astype(int)
counts = np.bincount(val)
print(counts)

plt.hist(counts.T, normed=True)
plt.show()



