import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neuro_helper.plot import savefig
import pandas as pd
from scipy.signal import welch, periodogram
from scipy.fft import fft

a = 1.1
x1 = np.arange(0, 30, 1).astype(float)
y1 = np.power(a, -x1)
x2 = x1[0::2]
y2 = np.power(a, -x2)

fig, ax = plt.subplots(figsize=(5, 5))
sns.lineplot(x=x1, y=y1, ax=ax, marker='o', linestyle="--")
sns.lineplot(x=x2, y=y2, ax=ax, marker='o', linestyle="--")
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)
# ax.axis([-np.pi, np.pi, -1, 1])
ax.set(xticklabels=[], yticklabels=[])
ax.grid(False)
plt.show()
print(savefig(fig, "sampling_plot", low=True))

from colorednoise import powerlaw_psd_gaussian
beta = 1.1 # the exponent
x1 = np.arange(0, 512)
y1 = powerlaw_psd_gaussian(beta, len(x1))
x2 = x1[::4]
y2 = y1[::4]
plt.figure()
sns.lineplot(x=x1, y=y1, marker='o', linestyle="--", color='b')
plt.show()
plt.figure()
sns.lineplot(x=x2, y=y2, marker='o', linestyle="--", color='r')
plt.show()


f1 = fft(y1)
f2 = fft(y2)
plt.figure()
sns.lineplot(x=x1, y=np.real(f1), marker='o', linestyle="--", color='black')
plt.show()
plt.figure()
sns.lineplot(x=x2, y=np.real(f2), marker='o', linestyle="--", color='green')
plt.show()


x1 = np.arange(-5, 5)
y1 = [0, 2, 3, 4, 2, 4, 1.5, 2, 0.75, 2]
x2 = x1[::2]
y2 = y1[::2]
x3 = x1[::4]
y3 = y1[::4]

plt.figure()
ax = sns.lineplot(x=x1, y=y1, marker='o', linestyle="--", color='black')
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)
# ax.axis([-np.pi, np.pi, -1, 1])
ax.set(xticklabels=[], yticklabels=[])
ax.grid(False)
plt.show()