# Autor: Daniel Iturralde
# Fecha: 29/01/2024
# VERSION 3.0 (revisada)
# Ejercicio Final - Estudio sobre precios de móviles con técnicas de Machine Learning
#
# Este script responde, en orden, a los 15 apartados planteados en el README.md
# del proyecto. Las figuras se guardan como archivos .png en la carpeta
# "figuras/" (en lugar de abrir ventanas con plt.show(), para que el script
# pueda ejecutarse también en entornos sin pantalla). Si lo ejecutas en tu
# propio ordenador y prefieres ver las ventanas emergentes, puedes sustituir
# las llamadas a guardar_figura(...) por plt.show().

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# HDBSCAN no es una dependencia estándar de scikit-learn y en algunos entornos
# no está instalada (o falla su compilación). La hacemos opcional para que el
# resto del script no se rompa si no está disponible.
try:
    import hdbscan
    HDBSCAN_DISPONIBLE = True
except ImportError:
    HDBSCAN_DISPONIBLE = False
    print("Aviso: el paquete 'hdbscan' no está instalado. Se omitirá ese apartado.")
    print("Para instalarlo: pip install hdbscan\n")

CARPETA_FIGURAS = "figuras"
os.makedirs(CARPETA_FIGURAS, exist_ok=True)


def guardar_figura(nombre):
    """Guarda la figura activa de matplotlib en la carpeta de figuras y la cierra."""
    ruta = os.path.join(CARPETA_FIGURAS, nombre)
    plt.savefig(ruta, dpi=120, bbox_inches="tight")
    print(f"  -> Figura guardada en: {ruta}")
    plt.close()


# ============================================================
# PASO 3: Cargar el CSV y explorar el DataFrame
# ============================================================
print("=" * 60)
print("PASO 3: Carga y exploración inicial del dataset")
print("=" * 60)

df = pd.read_csv("datos_moviles (2).csv")

print("\nPrimeras filas del DataFrame:")
print(df.head())

num_registros = len(df)
num_caracteristicas = df.shape[1]

print("\nNúmero de registros:", num_registros)
print("Número de características:", num_caracteristicas)


# ============================================================
# PASO 4: Correlaciones de todas las variables con price_range
# ============================================================
print("\n" + "=" * 60)
print("PASO 4: Correlaciones con price_range")
print("=" * 60)

correlations = df.corr()
price_range_correlation = correlations["price_range"].drop("price_range")
top_correlated_variables = price_range_correlation.abs().nlargest(5)

print("\nLas 5 variables con mayor correlación (en valor absoluto) con price_range son:")
print(top_correlated_variables)


# ============================================================
# PASO 5: Matriz de correlaciones entre price, ram y battery_power
# ============================================================
print("\n" + "=" * 60)
print("PASO 5: Matriz de correlaciones (price, ram, battery_power)")
print("=" * 60)

variables_corr_price = ["ram", "battery_power", "price"]
df_corr_price = df[variables_corr_price]
correlation_matrix = df_corr_price.corr()

print("\n", correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de correlaciones entre price, ram y battery_power", fontsize=14)
guardar_figura("05_matriz_correlaciones.png")


# ============================================================
# PASO 6 y 7: Regresión lineal de price frente a ram
# ============================================================
print("\n" + "=" * 60)
print("PASO 6-7: Regresión lineal price ~ ram")
print("=" * 60)

X = df[["ram"]]
y = df["price"]

modelo_reg = LinearRegression()
modelo_reg.fit(X, y)

coeficiente = modelo_reg.coef_[0]
intercepto = modelo_reg.intercept_
y_pred = modelo_reg.predict(X)
r2 = r2_score(y, y_pred)

print(f"\nCoeficiente de regresión (pendiente): {coeficiente:.4f}")
print(f"Término independiente (intercepto): {intercepto:.4f}")
print(f"Coeficiente de determinación R²: {r2:.4f}")
if r2 >= 0.7:
    valoracion = "un buen ajuste"
elif r2 >= 0.5:
    valoracion = "un ajuste moderado (mejorable añadiendo más variables)"
else:
    valoracion = "un ajuste pobre"
print(f"Interpretación: con R²={r2:.2f} se obtiene {valoracion}.")

# Gráfica de la recta de regresión sobre los datos
plt.figure(figsize=(8, 6))
plt.scatter(df["ram"], df["price"], alpha=0.4, s=15, label="Datos reales")
orden = np.argsort(df["ram"].values)
plt.plot(df["ram"].values[orden], y_pred[orden], color="red", linewidth=2,
         label=f"Recta de regresión (R²={r2:.2f})")
plt.title("Regresión lineal: price frente a ram", fontsize=14)
plt.xlabel("RAM (MB)")
plt.ylabel("Precio (€)")
plt.legend()
plt.grid(True)
guardar_figura("06_regresion_lineal.png")

# Predicción para 3100 MB de RAM
ram_prediccion = np.array([[3100]])
precio_predicho = modelo_reg.predict(ram_prediccion)
print(f"\nEl precio estimado para un móvil con 3100 MB de RAM es: {precio_predicho[0]:.2f} €")


# ============================================================
# PASO 8: Residuos frente a valores predichos
# ============================================================
print("\n" + "=" * 60)
print("PASO 8: Residuos del modelo de regresión")
print("=" * 60)

residuos = y - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuos, color="blue", alpha=0.5, s=15)
plt.title("Residuos vs. Valores Predichos", fontsize=14)
plt.xlabel("Valores Predichos (€)")
plt.ylabel("Residuos (€)")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlim(20, 2000)
plt.ylim(-800, 800)
plt.grid(True)
guardar_figura("08_residuos.png")


# ============================================================
# PASO 9: Clasificación con SVM, kernel lineal (ram, battery_power -> price_range)
# ============================================================
print("\n" + "=" * 60)
print("PASO 9: Clasificación SVM kernel lineal (OvR)")
print("=" * 60)

X_classification = df[["ram", "battery_power"]].values
price_range = np.array(df["price_range"])  # tal como pide el enunciado

X_train, X_test, y_train, y_test = train_test_split(
    X_classification, price_range, test_size=0.2, random_state=42
)

# Escalamos las variables antes de entrenar el SVM: ram y battery_power tienen
# rangos muy distintos, y sin escalar el solver del kernel lineal no converge
# bien (sklearn lo advierte explícitamente: "Consider pre-processing your data
# with StandardScaler or MinMaxScaler").
scaler_svm = StandardScaler()
X_train = scaler_svm.fit_transform(X_train)
X_test = scaler_svm.transform(X_test)

svm_lineal = SVC(kernel="linear", decision_function_shape="ovr")
svm_lineal.fit(X_train, y_train)
y_pred_lineal = svm_lineal.predict(X_test)
accuracy_lineal = accuracy_score(y_test, y_pred_lineal)
print(f"\nExactitud del test (SVM, kernel lineal, OvR): {accuracy_lineal:.4f}")


def graficar_regiones_decision(modelo, X, y_true, titulo, nombre_archivo):
    """Dibuja las regiones de decisión de un clasificador 2D (alternativa
    casera a mlxtend.plotting.plot_decision_regions, por si esa librería
    no está instalada)."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis",
                           edgecolor="k", s=20)
    plt.legend(*scatter.legend_elements(), title="price_range")
    plt.title(titulo, fontsize=14)
    plt.xlabel("ram (estandarizada)")
    plt.ylabel("battery_power (estandarizada)")
    guardar_figura(nombre_archivo)


# Intentamos usar mlxtend si está disponible; si no, usamos la función propia
try:
    from mlxtend.plotting import plot_decision_regions
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_train, y_train, clf=svm_lineal, legend=2)
    plt.title("Regiones de decisión - SVM kernel lineal", fontsize=14)
    plt.xlabel("ram")
    plt.ylabel("battery_power")
    guardar_figura("09_svm_lineal_regiones_mlxtend.png")
except ImportError:
    graficar_regiones_decision(
        svm_lineal, X_train, y_train,
        "Regiones de decisión - SVM kernel lineal",
        "09_svm_lineal_regiones.png",
    )


# ============================================================
# PASO 10: SVM con kernel de base radial gaussiana (RBF), gamma=20
# ============================================================
print("\n" + "=" * 60)
print("PASO 10: Clasificación SVM kernel RBF, gamma=20")
print("=" * 60)

svm_rbf = SVC(kernel="rbf", gamma=20, decision_function_shape="ovr")
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

print(f"\nExactitud del test (SVM, kernel RBF, gamma=20): {accuracy_rbf:.4f}")
print(f"Comparación: kernel lineal = {accuracy_lineal:.4f} | kernel RBF (gamma=20) = {accuracy_rbf:.4f}")
print("Con un valor de gamma tan alto, el modelo RBF sobreajusta fuertemente a los")
print("datos de entrenamiento (fronteras de decisión muy 'recortadas' alrededor de")
print("cada punto) y generaliza mal sobre el conjunto de test.")

graficar_regiones_decision(
    svm_rbf, X_train, y_train,
    "Regiones de decisión - SVM kernel RBF (gamma=20)",
    "10_svm_rbf_gamma20_regiones.png",
)


# ============================================================
# PASO 11: Estrategia OvR explícita (One-vs-Rest) y exactitud
# ============================================================
print("\n" + "=" * 60)
print("PASO 11: Estrategia explícita One-vs-Rest (OvR)")
print("=" * 60)

from sklearn.multiclass import OneVsRestClassifier

ovr_model = OneVsRestClassifier(SVC(kernel="linear"))
ovr_model.fit(X_train, y_train)
y_pred_ovr = ovr_model.predict(X_test)
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)

print(f"\nExactitud del test con OneVsRestClassifier explícito: {accuracy_ovr:.4f}")
print("(SVC con decision_function_shape='ovr' ya aplica internamente una estrategia")
print("equivalente; OneVsRestClassifier lo hace de forma explícita entrenando un")
print("clasificador binario por cada clase frente al resto.)")


# ============================================================
# PASO 12: K-medias sobre (ram, price) sin etiquetas
# ============================================================
print("\n" + "=" * 60)
print("PASO 12: Agrupamiento K-medias (ram, price)")
print("=" * 60)

X_clustering = df[["ram", "price"]].values
scaler = StandardScaler()
X_clustering_esc = scaler.fit_transform(X_clustering)

# Método del codo para elegir k
inertias = []
rango_k = range(1, 10)
for k in rango_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clustering_esc)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(list(rango_k), inertias, marker="o")
plt.title("Método del codo para K-medias", fontsize=14)
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Inercia")
plt.grid(True)
guardar_figura("12_kmeans_codo.png")

# A partir de la gráfica del codo se observa un cambio de pendiente claro
# alrededor de k=3-4. Comparamos k=3 frente a k=4 (k+1).
k_elegido = 3
for k in [k_elegido, k_elegido + 1]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas_km = km.fit_predict(X_clustering_esc)
    plt.figure(figsize=(8, 6))
    plt.scatter(df["ram"], df["price"], c=etiquetas_km, cmap="viridis", s=15)
    plt.title(f"K-medias con k={k} clústeres", fontsize=14)
    plt.xlabel("ram")
    plt.ylabel("price")
    guardar_figura(f"12_kmeans_k{k}.png")

print(f"\nSegún el método del codo, un número razonable de clústeres es k={k_elegido}.")
print(f"Se han generado y comparado las soluciones para k={k_elegido} y k={k_elegido + 1}.")


# ============================================================
# PASO 13: DBSCAN y HDBSCAN sobre (ram, price)
# ============================================================
print("\n" + "=" * 60)
print("PASO 13: Agrupamiento DBSCAN y HDBSCAN")
print("=" * 60)

# Gráfico de distancia al k-ésimo vecino más cercano (sobre datos
# estandarizados; sin estandarizar, ram y price tienen escalas tan distintas
# que un único epsilon no tiene sentido para ambas variables a la vez).
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_clustering_esc)
distancias, _ = nn.kneighbors(X_clustering_esc)
distancias = np.sort(distancias[:, 4])

plt.figure(figsize=(8, 6))
plt.plot(distancias)
plt.title("Gráfico de distancia al 5º vecino más cercano (datos estandarizados)", fontsize=14)
plt.xlabel("Puntos ordenados por distancia")
plt.ylabel("Distancia al k-ésimo vecino más cercano")
plt.grid(True)
guardar_figura("13_dbscan_kdistancias.png")

# Tras inspeccionar el "codo" de la gráfica anterior, eps=0.2 resulta razonable
epsilon = 0.2
dbscan_model = DBSCAN(eps=epsilon, min_samples=5)
dbscan_clusters = dbscan_model.fit_predict(X_clustering_esc)
n_clusters_dbscan = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
n_ruido_dbscan = list(dbscan_clusters).count(-1)

print(f"\nDBSCAN (eps={epsilon}, min_samples=5): {n_clusters_dbscan} clústeres, "
      f"{n_ruido_dbscan} puntos de ruido")

plt.figure(figsize=(8, 6))
plt.scatter(df["ram"], df["price"], c=dbscan_clusters, cmap="viridis", s=15)
plt.title(f"DBSCAN (eps={epsilon}) sobre ram y price", fontsize=14)
plt.xlabel("ram")
plt.ylabel("price")
guardar_figura("13_dbscan_resultado.png")

# HDBSCAN (solo si el paquete está disponible)
if HDBSCAN_DISPONIBLE:
    hdbscan_model = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=20)
    hdbscan_clusters = hdbscan_model.fit_predict(X_clustering_esc)
    n_clusters_hdbscan = len(set(hdbscan_clusters)) - (1 if -1 in hdbscan_clusters else 0)
    n_ruido_hdbscan = list(hdbscan_clusters).count(-1)

    print(f"HDBSCAN (min_samples=10, min_cluster_size=20): {n_clusters_hdbscan} clústeres, "
          f"{n_ruido_hdbscan} puntos de ruido")

    plt.figure(figsize=(8, 6))
    plt.scatter(df["ram"], df["price"], c=hdbscan_clusters, cmap="viridis", s=15)
    plt.title("HDBSCAN sobre ram y price", fontsize=14)
    plt.xlabel("ram")
    plt.ylabel("price")
    guardar_figura("13_hdbscan_resultado.png")
else:
    print("HDBSCAN omitido (paquete no instalado).")


# ============================================================
# PASO 14: Agrupamiento por aglomeración (jerárquico)
# ============================================================
print("\n" + "=" * 60)
print("PASO 14: Agrupamiento por aglomeración")
print("=" * 60)

num_clusters_aggl = 4
aggl_model = AgglomerativeClustering(n_clusters=num_clusters_aggl)
aggl_clusters = aggl_model.fit_predict(X_clustering_esc)

plt.figure(figsize=(8, 6))
plt.scatter(df["ram"], df["price"], c=aggl_clusters, cmap="viridis", s=15)
plt.title(f"Agrupamiento por aglomeración (k={num_clusters_aggl})", fontsize=14)
plt.xlabel("ram")
plt.ylabel("price")
guardar_figura("14_aglomerativo_resultado.png")

print(f"\nSe ha aplicado clustering aglomerativo con {num_clusters_aggl} clústeres.")
print("Tamaño de cada clúster:", np.bincount(aggl_clusters))


# ============================================================
# PASO 15: PCA sobre el dataset completo
# ============================================================
print("\n" + "=" * 60)
print("PASO 15: Análisis de Componentes Principales (PCA)")
print("=" * 60)

# Se excluye price_range por ser una variable categórica derivada de price
# (incluirla introduciría una fuga de información redundante). Se estandariza
# el resto de variables porque están en escalas muy distintas (p. ej. ram en
# miles frente a variables binarias 0/1).
df_pca = df.drop(columns=["price_range"])
X_pca_esc = StandardScaler().fit_transform(df_pca)

pca = PCA()
pca.fit(X_pca_esc)
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)
num_dimensiones_95 = np.argmax(varianza_acumulada >= 0.95) + 1

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(varianza_explicada) + 1), varianza_acumulada,
         marker="o", linestyle="-", color="b")
plt.title("Varianza explicada acumulada por número de dimensiones", fontsize=14)
plt.xlabel("Número de dimensiones")
plt.ylabel("Varianza explicada acumulada")
plt.axvline(x=num_dimensiones_95, color="r", linestyle="--",
            label=f"95% de varianza (dim={num_dimensiones_95})")
plt.legend()
plt.grid(True)
guardar_figura("15_pca_varianza.png")

print(f"\nNúmero de dimensiones requeridas para preservar el 95% de la varianza: "
      f"{num_dimensiones_95} (de {df_pca.shape[1]} variables originales)")


print("\n" + "=" * 60)
print("Script finalizado. Todas las figuras se han guardado en la carpeta "
      f"'{CARPETA_FIGURAS}/'.")
print("=" * 60)
