import numpy as np
import torch
import matplotlib.pyplot as plt
X = torch.tensor(np.linspace(-7,7, 1000))
y = torch.sigmoid(X)

plt.rcParams['text.usetex'] = True
plt.plot(X,y, linewidth=1.5, color='blue')

plt.axvline(x = -4.472, color='r', linestyle='--')
plt.text(4.8, 0.2, r'$\beta^T X_i = 4.472$', ha='left', va='center', rotation=90, color='red')
plt.text(-5, 0.2, r'$\beta^T X_i = -4.472$', ha='left', va='center', rotation=90, color='red')

plt.axvline(x = 4.472, color='r', linestyle='--')
plt.xlim([-6,6])
plt.xlabel(r'$\beta^T X_i$')
plt.ylabel(r'Sigmoid($\beta^TX_i$)')
plt.title(r'Plot showing the range of $\beta^TX_i $ after scaling')

plt.grid()
plt.savefig('demo_plotting/plots/sigmoid.pdf')
plt.savefig('demo_plotting/plots/sigmoid.jpg')