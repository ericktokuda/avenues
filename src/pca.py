import numpy as np
import pandas as pd
import sys
from myutils import transform
import matplotlib.pyplot as plt

featpath = './data/features.csv'
df = pd.read_csv(featpath)
cols = df.columns[1:]
data = df[cols].values

tr, evecs, vals = transform.pca(data, normalize=True)
pcs, contribs = transform.get_pc_contribution(evecs)

W = 640; H = 480
fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

cities = []
for c in list(df.city):
    abbr = c.upper()[:3]
    if abbr == 'LIV': abbr = 'LV' + c[3].upper()
    cities.append(abbr)

ax.scatter(tr[:, 0], tr[:, 1])

for i in range(len(df.city)):
    ax.annotate(cities[i], (tr[i, 0], tr[i, 1]))

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()

outpath = '/tmp/pca.png'
plt.savefig(outpath)
