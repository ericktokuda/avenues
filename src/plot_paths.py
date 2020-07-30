import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os; from os.path import join as pjoin

outdir = '/tmp/out/'
results = '/tmp/out/results.csv'
plotpath = pjoin(outdir, 'paths.png')

df = pd.read_csv(results)
plt.scatter(df.nbridges, df.avgpathlen)
sigma = np.std(df.avgpathlen)
plt.ylim(np.min(df.avgpathlen) - sigma, np.max(df.avgpathlen) + sigma)
plt.xlabel('Number of bridges')
plt.ylabel('Mean of shortest path length')
plt.savefig(plotpath)
print('Please check ' + plotpath)
