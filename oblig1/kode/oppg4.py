import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def q(x, y, A):
    f = y + A*np.sin(np.pi*y)*(1 - x)
    return f

def g(y):
    f = -np.sin(np.pi*y)
    return f

def psimean(x, y):
    f = g(y)*(x - 1)
    return f

def qmean(x, y):
    f = y + psimean(x, y)
    return f

def psi1_blocked(x, y):
    f = (x - 1)*g(y)
    return f

def psi2_blocked(x, y):
    f = np.zeros((x.shape))
    return f

def psi1_closed(x, y):
    f = psimean(x, y) - psi2_closed(x, y)
    return f

def psi2_closed(x, y):
    f = 0.5*(qmean(x, y) - 1)
    return f

def q1_blocked(x, y):
    f = y + psi2_blocked(x, y) - psi1_blocked(x, y)
    return f

def q2_blocked(x, y):
    f = y + psi1_blocked(x, y) - psi2_blocked(x, y)
    return f

def q1_closed(x, y):
    f = 2*y - 1
    return f

def q2_closed(x, y):
    f = np.ones((x.shape))
    return f



sns.set()
sns.set_style("white")
sns.set_palette("Set2")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

figdir = "../figurar/"


N = 1000
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

X, Y = np.meshgrid(x, y)

As = [0.25, 0.50, 1.00]
fig, axes = plt.subplots(1, 3, figsize=(10, 8), sharey=True)
for ax, A in zip(axes, As):
    ax.contour(X, Y, q(X, Y, A), colors="black", linewidths=2, levels=20)
    ax.contour(X, Y, q(X, Y, A) - 1, levels=0, colors="red", linewidths=2, linestyles="dashed")
    ax.set_xlabel("X", fontsize=14)
    ax.set_title(f"A = {A}", fontsize=18)
    ax.set_aspect("equal")
axes[0].set_ylabel("Y", fontsize=14)
fig.tight_layout()
fig.savefig(figdir + "oppg4_2.pdf")


fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
names = np.array([[r"$q_1$", r"$q_2$"], [r"$\psi_1$", r"$\psi_2$"]])
closed = np.array([[q1_closed(X, Y), q2_closed(X, Y)], [psi1_closed(X, Y), psi2_closed(X, Y)]])
blocked = np.array([[q1_blocked(X, Y), q2_blocked(X, Y)], [psi1_blocked(X, Y), psi2_blocked(X, Y)]])
for row in range(axes.shape[0]):
    for col in range(axes.shape[1]):
        axes[row, col].contour(X, Y, np.ma.masked_where(qmean(X, Y) < 1, closed[row, col]),
                                colors="black", linewidths=2, levels=8)
        axes[row, col].contour(X, Y, np.ma.masked_where(qmean(X, Y) > 1, blocked[row, col]),
                                colors="black", linewidths=2, levels=8)
        axes[row, col].contour(X, Y, qmean(X, Y) - 1,
                            levels=0, colors="red", linewidths=2, linestyles="dashed"
                            )
        axes[1, col].set_xlabel("X", fontsize=14)
        axes[row, col].set_title(names[row, col], fontsize=18)
    axes[row, 0].set_ylabel("Y", fontsize=14)
fig.tight_layout()
fig.savefig(figdir + "oppg4_1.pdf")
plt.show()
