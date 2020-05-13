import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
plt.style.use('seaborn')

df = pd.DataFrame({'Peso': np.array([40.0, 93.5, 35.5, 30.0, 52.0, 17.0, 38.5,  8.5, 33.0,  9.5, 21.0, 79.0]),
                    'Altura': np.array([42.8, 63.5, 37.5, 39.5, 45.5, 38.5, 43.0, 22.5, 37.0, 23.5, 33.0, 58.0]),
                    'Cateter': [37, 50, 34, 36, 43, 28, 37, 20, 34, 30, 38, 47]})


def data_analisis_linreg(caminho):
    frame = pd.read_excel(caminho)
    p = frame.iloc[:, 1]


def graf():
    # plt.show()
    print(f"EQUATION: Theta(x) = {a}x + {b} , R² = {R2}")

X = np.array(df[["Peso"]])
y = np.array(df[["Altura"]])
reg = linear_model.LinearRegression()
reg.fit(X, y)
R2 = reg.score(X, y)
a = reg.coef_
b = reg.intercept_
y_pred = reg.predict(X)

plt.scatter(X, y, color='black', marker ="x", s = 100)
plt.plot(X,y_pred, color='blue', linewidth=3, label ="Linha de predição")
plt.legend()
plt.title("Linear Regression")
plt.xlabel("Peso")
plt.ylabel("Altura")
plt.grid()
plt.show()

graf()