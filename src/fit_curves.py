import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def linearf(x, a, b):
    y = a*x + b
    return y

def powerf(x, a, b):
    y = a*np.power(x, b)
    return y

def expf(x, a, b):
    y = a*np.exp(b * x)
    return y

df = pd.read_csv('./data/lacun21_data.csv')


citylabels = np.array([
'BRI',
'CAS',
'EMD',
'EST',
'GAP',
'GEL',
'GEI',
'IST',
'JEN',
'KAS',
'KOB',
'LVE',
'LVO',
'PAR',
'SAN',
'VIL',
'WIL',
'WOR'])



x = (df.lacun.values)
y = (df.gain.values)
inds = np.argsort(x)

x = np.array(list(sorted(x)))
y = y[inds]
lbls = citylabels[inds]

W = 640; H = 480
fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

for i in range(len(x)):
    ax.scatter((x[i]), (y[i]), c='#1f77b4')
    ax.annotate(lbls[i], [(x[i]), (y[i])])


ax.set_xlabel('Lacunarity');
ax.set_ylabel('Gain');

# ret = optimize.curve_fit(linearf, xdata = np.log(x), ydata = np.log(y))
ret = optimize.curve_fit(powerf, xdata = (x), ydata = (y))
coeffs = ret[0]
errs = np.sqrt(np.diag(ret[1]))
print('Linear fit of the log values:', coeffs, errs)
yfit = powerf(x, coeffs[0], coeffs[1])
ax.plot(x, yfit, c='orange', label='Linear')

ax.ticklabel_format(style='plain')
ax.set_xscale("log")
ax.set_yscale("log")
# plt.xticks(rotation=45)

# ret = optimize.curve_fit(powerf, xdata = x, ydata = y)
# coeffs = ret[0]
# errs = np.sqrt(np.diag(ret[1]))
# print('Power law fit of the orig values:', coeffs, errs)
# yfit = powerf(x, coeffs[0], coeffs[1])
# plt.plot(x, yfit, c='orange', label='Power')

# ret = optimize.curve_fit(expf, xdata = x, ydata = y)
# coeffs = ret[0]
# errs = np.sqrt(np.diag(ret[1]))
# print(coeffs, errs)
# yfit = expf(x, coeffs[0], coeffs[1])
# plt.plot(x, yfit, label='Exp')

# plt.legend()
# plt.legend()
plt.savefig('/tmp/fits.png')


