df = pd.DataFrame({'Peso': np.array([18.141,42.404,16.1,13.005,23.583,7.710,17.46,3.855,14.966,4.308,9.524,35.828]),
                    'Altura': np.array([1.087,1.613,0.953,1.003,1.156,0.978,1.092,0.572,0.940,0.597,0.838,1.473]),
                    'Cateter': np.array([37, 50, 34, 36, 43, 28, 37, 20, 34, 30, 38, 47])})

X = np.ones(24).reshape(12,2)
X[:,1] = df["Altura"]
y = np.array(df["Cateter"]).reshape((12,1))
X = np.matrix(X)
y = np.matrix(y)
XT = X.T
w = np.linalg.inv((XT*X))*XT*y
w

print(f"Theta(x) = {w[1]}x + {w[0]}")
