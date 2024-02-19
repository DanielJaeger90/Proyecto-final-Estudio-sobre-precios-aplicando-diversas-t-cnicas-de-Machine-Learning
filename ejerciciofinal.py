# Autor: Daniel Iturralde
# Fecha: 29/01/2024
# VERSION 2.0
# Ejercicio Final

# Importamos librerias necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import hdbscan

# Paso 1: Cargar el archivo CSV en un DataFrame
# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('datos_moviles (2).csv')

# Visualizar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Obtener el número de registros y características
num_registros = len(df)
num_caracteristicas = df.shape[1]

print("\nNúmero de registros:", num_registros)
print("Número de características:", num_caracteristicas)

#Paso2 : Obtener el número de registros y características

# Calcular las correlaciones entre todas las variables
correlations = df.corr()

# Obtener las correlaciones con respecto a price_range
price_range_correlation = correlations['price_range']

# Excluir la correlación de price_range consigo misma
price_range_correlation = price_range_correlation.drop('price_range')

# Obtener las cinco variables con mayor correlación absoluta con price_range
top_correlated_variables = price_range_correlation.abs().nlargest(5)

print("Las 5 variables con mayor correlación con price_range son:")
print(top_correlated_variables)

# Pase 3  : Graficar matriz de correlaciones para ram y battery_power

# Seleccionar las variables más correlacionadas con price
variables_corr_price = ['ram', 'battery_power']

# Incluir la variable price en la lista de variables
variables_corr_price.append('price')

# Filtrar el DataFrame para seleccionar solo las columnas de interés
df_corr_price = df[variables_corr_price]

# Calcular la matriz de correlaciones
correlation_matrix = df_corr_price.corr()

# Generar el heatmap de la matriz de correlaciones
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de correlaciones entre price, ram y battery_power', fontsize=14)
plt.show()
# Paso 4: Obtener las correlaciones con respecto a price_range
correlations = df.corr()
price_range_correlation = correlations['price_range'].drop('price_range')
top_correlated_variables = price_range_correlation.abs().nlargest(5)
print("Las 5 variables con mayor correlación con price_range son:")
print(top_correlated_variables)

# Paso 5: Graficar matriz de correlaciones para ram, battery_power y price
variables_corr_price = ['ram', 'battery_power', 'price']
df_corr_price = df[variables_corr_price]
correlation_matrix = df_corr_price.corr()
plt.figure(figsize=(8, 6))
plt.title('Matriz de correlaciones entre price, ram y battery_power', fontsize=14)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.show()

# Paso 6: Regresión lineal de price frente a ram
from sklearn.linear_model import LinearRegression
X = df[['ram']]
y = df['price']
model = LinearRegression()
model.fit(X, y)
ram_prediccion = np.array([[3100]])
precio_predicho = model.predict(ram_prediccion)
print("El precio estimado para un móvil con 3100 MB de RAM es:", precio_predicho[0], "€")

# Paso 7: Representar gráficamente los residuos
residuos = y - model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(model.predict(X), residuos, color='blue')
plt.title('Residuos vs. Valores Predichos', fontsize=14)
plt.xlabel('Valores Predichos (€)')
plt.ylabel('Residuos (€)')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlim(20, 2000)
plt.ylim(-800, 800)  
plt.grid(True)
plt.show()

# Paso 9: Clasificación OvR con SVM y kernel lineal
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X_classification = df[['ram', 'battery_power']]
X_train, X_test, y_train, y_test = train_test_split(X_classification, df['price_range'], test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Exactitud del test (OvR SVM con kernel lineal):", accuracy)

# Paso 13: Clasificación con DBSCAN y HDBSCAN
X_clustering = df[['ram', 'price']]
# Método DBSCAN
nn = NearestNeighbors(n_neighbors=2)
nn.fit(X_clustering)
distances, indices = nn.kneighbors(X_clustering)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(8, 6))
plt.plot(distances)
plt.title('Gráfico de distancia k más cercano', fontsize=14)
plt.xlabel('Puntos ordenados por distancia')
plt.ylabel('Distancia al k-ésimo vecino más cercano')
plt.grid(True)
plt.show()
epsilon = 0.3
dbscan_model = DBSCAN(eps=epsilon)
dbscan_clusters = dbscan_model.fit_predict(X_clustering)

# Método HDBSCAN
hdbscan_model = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=20)
hdbscan_clusters = hdbscan_model.fit_predict(X_clustering)

# Paso 14: Agrupamiento por aglomeración
num_clusters_aggl = 4  
aggl_model = AgglomerativeClustering(n_clusters=num_clusters_aggl)
aggl_clusters = aggl_model.fit_predict(X_clustering)

# Paso 15: Aplicar PCA y obtener la varianza explicada
pca = PCA()
pca.fit(df)
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)
num_dimensiones_95 = np.argmax(varianza_acumulada >= 0.95) + 1

# Graficar la varianza explicada en función del número de dimensiones
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(varianza_explicada) + 1), varianza_acumulada, marker='o', linestyle='-', color='b')
plt.title('Varianza explicada por número de dimensiones', fontsize=14)
plt.xlabel('Número de dimensiones')
plt.ylabel('Varianza explicada acumulada')
plt.axvline(x=num_dimensiones_95, color='r', linestyle='--', label=f'95% de varianza (dim={num_dimensiones_95})')
plt.legend()
plt.grid(True)
plt.show()
print(f"Número de dimensiones requeridas para preservar una varianza del 95%: {num_dimensiones_95}")
