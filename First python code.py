# %% SECCION 1
a=1
b=2  
c=a*b


d=4
print(d)
# Comento aquí lo que sea

# %% SECCION 2 - Numpy basico
import numpy as np

# Un vector es simplemente un array de una dimension
v1 = np.array([1, 2, 3, 4, 5])
print(v1)
print(type(v1))

# %% Vectores con rangos
v2 = np.arange(0, 10)       # de 0 a 9
v3 = np.arange(0, 10, 2)    # de 0 a 9, saltando de 2 en 2
v4 = np.linspace(0, 1, 5)   # 5 numeros igualmente espaciados entre 0 y 1

print(v2)
print(v3)
print(v4)

# %% Vectores de ceros, unos y aleatorios
v5 = np.zeros(5)
v6 = np.ones(5)
v7 = np.random.rand(5)  # 5 numeros aleatorios entre 0 y 1

print(v5)
print(v6)
print(v7)

# %% Operaciones basicas con vectores
v8 = np.array([1, 2, 3])
v9 = np.array([10, 20, 30])

print(v8 + v9)   # suma elemento a elemento
print(v8 * v9)   # multiplicacion elemento a elemento
print(v8.sum())  # suma de todos los elementos
print(v8.mean()) # media
print(v8 * 2)    # multiplicar por un escalar

# %% SECCION 3 - Mas estadisticas y producto vectorial
print(v8.std())    # desviacion estandar
print(v8.max())    # valor maximo
print(v8.min())    # valor minimo
print(v8.argmax()) # indice del valor maximo
print(v8.argmin()) # indice del valor minimo
print(np.dot(v8, v9))  # producto escalar (dot product)

# %% SECCION 4 - Indexado y slicing
v10 = np.arange(10, 20)  # [10, 11, ..., 19]
print(v10)
print(v10[0])     # primer elemento
print(v10[-1])    # ultimo elemento
print(v10[2:5])   # slice: del indice 2 al 4
print(v10[::2])   # de dos en dos
print(v10[::-1])  # invertido

# %% SECCION 5 - Indexado booleano (filtrado)
v11 = np.array([3, -1, 7, 0, -5, 9])
print(v11 > 0)          # array de booleanos
print(v11[v11 > 0])     # solo los valores positivos
v11[v11 < 0] = 0        # reemplaza negativos por 0
print(v11)

# %% SECCION 6 - Matrices (arrays 2D)
m1 = np.array([[1, 2, 3], [4, 5, 6]])
print(m1)
print(m1.shape)    # (filas, columnas)
print(m1.T)        # traspuesta
print(m1.reshape(3, 2))  # cambia la forma sin cambiar los datos

m2 = np.array([[1, 0], [0, 1]])
m3 = np.array([[2, 3], [4, 5]])
print(m2 @ m3)      # multiplicacion de matrices (tambien: np.matmul(m2, m3))

# %% SECCION 7 - Combinar y ordenar arrays
v12 = np.array([5, 3, 1])
v13 = np.array([2, 4, 6])
print(np.concatenate([v12, v13]))  # unir arrays
print(np.sort(v12))                # orden ascendente
print(np.unique([1, 1, 2, 3, 3]))  # valores unicos

# %% SECCION 8 - Funciones matematicas universales (ufuncs)
v14 = np.array([1, 4, 9, 16])
print(np.sqrt(v14))   # raiz cuadrada elemento a elemento
print(np.exp(v8))     # exponencial
print(np.log(v14))    # logaritmo natural
print(np.round(np.array([1.234, 5.678]), 1))  # redondeo a 1 decimal

# %% SECCION 9 - Diferencias y acumulados (utiles para series de tiempo)
v15 = np.array([10, 12, 15, 14, 20])
print(np.diff(v15))    # diferencia entre elementos consecutivos
print(np.cumsum(v15))  # suma acumulada

# %% SECCION 10 - Pandas: los 10 comandos mas usados
# Pandas sirve para manipular datos en tablas (DataFrames), como una hoja de
# Excel: leer/escribir archivos, filtrar, transformar, agrupar y resumir datos.
import pandas as pd

data = {
    "dia": ["lun", "mar", "mie", "jue", "vie"],
    "demanda_MW": [120, 135, 128, 150, 142],
    "precio_EUR": [45.5, 48.2, 46.1, 52.0, 49.8],
}

# 1. pd.DataFrame() - crear una tabla a partir de datos
df = pd.DataFrame(data)

# 2. df.head() - ver las primeras filas (tambien existe df.tail())
print(df.head())

# 3. df.info() - resumen de columnas, tipos y valores nulos
print(df.info())

# 4. df.describe() - estadisticas descriptivas de las columnas numericas
print(df.describe())

# 5. df["columna"] / df[["c1", "c2"]] - seleccionar columnas
print(df["demanda_MW"])

# 6. df.loc[] / df.iloc[] - seleccionar filas por etiqueta o por posicion
print(df.loc[df["demanda_MW"] > 130])  # filtrado condicional
print(df.iloc[0:2])                    # por posicion

# 7. df["nueva"] = ... - crear/transformar columnas
df["ingreso_EUR"] = df["demanda_MW"] * df["precio_EUR"]

# 8. df.sort_values() - ordenar filas segun una columna
print(df.sort_values("demanda_MW", ascending=False))

# 9. df.groupby() - agrupar y agregar datos
df["semana"] = ["S1", "S1", "S1", "S2", "S2"]
print(df.groupby("semana")["demanda_MW"].mean())

# 10. pd.read_csv() / df.to_csv() - leer y guardar datos
# df.to_csv("demanda.csv", index=False)
# df_leido = pd.read_csv("demanda.csv")

# %% SECCION 11 - Pandas: algunos comandos adicionales utiles

# df.rename() - renombrar columnas
df = df.rename(columns={"demanda_MW": "demanda_mw"})

# df["col"].apply() - aplicar una funcion a cada valor de una columna
df["precio_categoria"] = df["precio_EUR"].apply(lambda x: "alto" if x > 48 else "bajo")

# df["col"].value_counts() - contar valores unicos
print(df["precio_categoria"].value_counts())

# df.isnull().sum() - contar valores nulos por columna
print(df.isnull().sum())

# df.pivot_table() - tabla dinamica con agregacion
print(df.pivot_table(values="demanda_mw", index="semana", aggfunc="mean"))

# pd.merge() - combinar dos DataFrames por una columna comun
info_dias = pd.DataFrame({
    "dia": ["lun", "mar", "mie", "jue", "vie"],
    "tipo_dia": ["laborable"] * 5,
})
df_merged = pd.merge(df, info_dias, on="dia")
print(df_merged)
