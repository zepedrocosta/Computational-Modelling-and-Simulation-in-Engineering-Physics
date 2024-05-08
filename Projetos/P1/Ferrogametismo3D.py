import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def transitionFunctionValues(t, h=0):
    # Adjusted for 3D
    deltaE = [[[k + j * h + i * h * h for k in range(-1, 3, 2)] for j in range(-4, 6, 2)] for i in range(-4, 6, 2)]
    output = [
        [[1 if elem <= 0 else np.exp(-2 * elem / t) for elem in row] for row in plane] for plane in deltaE
    ]
    return np.array(output)

def w(sigma, sigSoma, valuesW):
    # Adjusted for 3D
    i = int(sigSoma / 2 + 2)
    j = int(sigma / 2 + 1 / 2)
    k = int((sigSoma - sigma) / 2 + 2)    
    i = min(max(i, 0), valuesW.shape[0] - 1)
    j = min(max(j, 0), valuesW.shape[1] - 1)
    k = min(max(k, 0), valuesW.shape[2] - 1)
    return valuesW[i, j, k]

def vizinhosTabela(tamanho):
    # Adjusted for 3D
    vizMais = [i + 1 for i in range(tamanho)]
    vizMais[-1] = 0
    vizMenos = [i - 1 for i in range(tamanho)]
    return np.array([vizMais, vizMenos])

def inicializacao(tamanho, valor=-1):
    # Adjusted for 3D
    if valor != 0:
        rede = np.full((tamanho, tamanho, tamanho), valor, dtype="int")
    else:
        rede = np.random.choice([-1, 1], size=(tamanho, tamanho, tamanho))
    return rede

def cicloFerro(rede, tamanho, vizinhos, valuesW, h):
    e = 0
    for i in range(tamanho):
        for j in range(tamanho):
            for k in range(tamanho):
                sigma = rede[i, j, k]
                soma = (
                    rede[vizinhos[0, i], j, k]
                    + rede[vizinhos[1, i], j, k]
                    + rede[i, vizinhos[0, j], k]
                    + rede[i, vizinhos[1, j], k]
                    + rede[i, j, vizinhos[0, k]]
                    + rede[i, j, vizinhos[1, k]]
                )
                soma *= sigma
                etmp = -0.5 * (soma - sigma * h)
                p = np.random.random()
                if p < w(sigma, soma, valuesW):
                    rede[i, j, k] = -sigma
                    etmp = -etmp
                e += etmp
    return rede, e

def ferroSimul(tamanho, nCiclos, temp, h):
    rede = inicializacao(tamanho)
    valuesW = transitionFunctionValues(temp, h)
    vizinhos = vizinhosTabela(tamanho)
    order = np.zeros(nCiclos)
    e = np.zeros(nCiclos)
    for i in range(nCiclos):
        print("Ciclo", i)
        rede, eCiclo = cicloFerro(rede, tamanho, vizinhos, valuesW, h)
        order[i] = 2 * rede[rede == 1].shape[0] - tamanho**3
        e[i] = eCiclo
    order /= tamanho**3
    e /= tamanho**3
    return rede, order, e

t = 5.5
h = 0.0
tamanho = 10
rede, order, e = ferroSimul(tamanho, 10000, t, h)

# Visualizing the final state
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = np.where(rede == 1)
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

plt.plot(order)
plt.show()
