##
#Gradiente Estocastico
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

ind = np.array([40.0, 93.5, 35.5, 30.0, 52.0, 17.0, 38.5,  8.5, 33.0,  9.5, 21.0, 79.0])
dep = np.array([42.8, 63.5, 37.5, 39.5, 45.5, 38.5, 43.0, 22.5, 37.0, 23.5, 33.0, 58.0])
u = np.array([1,2,3,4,5])
v = np.array([5,7,9,11,13])
x = np.array([1 ,2 ,3, 4, 5])
y = np.array([2, 4, 6, 8,10])


def gradiente_descendente_estocastico(x, y, Custo=None):
    wo = w = 0
    inter = 10000
    alpha = 0.02  # learning rate
    n = len(x)
    custo = custot = np.array([])
    for t in range(inter):
        y_pred = x * w + wo
        custot = ((1 / n) * sum([val ** 2 for val in (y - y_pred)]))
        custo = np.append(custo, custot)
        wo = wo + alpha * (y - y_pred)
        w = w + alpha * (y - y_pred) * x

    plt.scatter(x, y, marker="x", color="black")
    plt.plot(x, y_pred, color="red")
    plt.xlabel("Independent variable")
    plt.ylabel("Dependent Variable")
    plt.show()
    print(f"Theta(x) = {w[-1]}x + {wo[-1]}")

    if Custo:
        plt.plot(np.array([t for t in range(0, inter)]), custo, color="orange")
        plt.xlabel("Iteration")
        plt.ylabel("Cost function")
        plt.show()


gradiente_descendente_estocastico(x, y, Custo=True)