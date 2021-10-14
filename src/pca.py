import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/frodo/projects/myutils/')
from myutils import transform
import matplotlib.pyplot as plt

featpath = '/home/frodo/results/avenues/20210729-results/features.csv'
df = pd.read_csv(featpath)
cols = df.columns[1:]
data = df[cols].values

tr, evecs, vals = transform.pca(data, normalize=True)
pcs, contribs = transform.get_pc_contribution(evecs)

W = 640; H = 480
fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
# ax.set_title('PCA components (pc1 and pc2)')

cities = [ c.capitalize() for c in list(df.city)]

ax.scatter(tr[:, 0], tr[:, 1])
# for i in range(tr.shape[0]):
    # ax.scatter(tr[i, 0], tr[i, 1], label=cities[i])

for i in range(len(df.city)):
    ax.annotate(cities[i], (tr[i, 0], tr[i, 1]))

xylim = np.max(np.abs(tr[:, 0:2])) * 1.1
# ax.set_xlabel('PCA1 ({} ({}%)'.format(cols[pcs[0]], int(contribs[0] * 100)))
# ax.set_ylabel('PCA2 {}: ({}%)'.format(cols[pcs[1]], int(contribs[1] * 100)))
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_xlim(-.2, +.65)
# ax.set_ylim(-xylim, +xylim)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
# plt.legend()

outpath = '/tmp/pca.png'
plt.savefig(outpath)
