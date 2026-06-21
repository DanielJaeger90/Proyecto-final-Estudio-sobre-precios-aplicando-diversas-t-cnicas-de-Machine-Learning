# Estudio sobre precios de móviles aplicando técnicas de Machine Learning

Proyecto final del examen de Machine Learning de CEINPRO. Analiza un dataset de
2.000 modelos de teléfono móvil para entender qué características determinan
su precio, aplicando varias técnicas de aprendizaje supervisado y no
supervisado: correlaciones, regresión lineal, clasificación con SVM,
clustering (K-medias, DBSCAN, HDBSCAN, aglomerativo) y reducción de
dimensionalidad con PCA.

## Planteamiento

Te pones en el lugar del departamento de marketing de una compañía de
telefonía móvil que ha recopilado datos de numerosos modelos de teléfono. El
objetivo es entender qué variables (RAM, batería, resolución de pantalla,
etc.) influyen en el precio de venta, para poder fijar precios razonables en
futuros modelos.

## Contenido del repositorio

```
.
├── ejerciciofinal.py        # Script principal con las 15 partes del estudio
├── datos_moviles (2).csv    # Dataset (2000 móviles, 22 columnas)
├── README.md                # Este documento
├── pyvenv.cfg                # Configuración del entorno virtual original (referencia)
└── figuras/                  # Se genera al ejecutar el script; contiene las gráficas
```

## El dataset

`datos_moviles (2).csv` contiene 2.000 registros con 22 columnas:

| Columna | Descripción |
|---|---|
| `battery_power` | Capacidad de la batería (mAh) |
| `blue` | Tiene Bluetooth (0/1) |
| `clock_speed` | Velocidad del microprocesador |
| `dual_sim` | Soporta doble SIM (0/1) |
| `fc` | Megapíxeles de la cámara frontal |
| `four_g` | Soporta 4G (0/1) |
| `int_memory` | Memoria interna (GB) |
| `m_dep` | Profundidad del móvil (cm) |
| `mobile_wt` | Peso del móvil (g) |
| `n_cores` | Número de núcleos del procesador |
| `pc` | Megapíxeles de la cámara principal |
| `px_height` / `px_width` | Resolución de pantalla en píxeles |
| `ram` | Memoria RAM (MB) |
| `sc_h` / `sc_w` | Alto y ancho de la pantalla (cm) |
| `talk_time` | Autonomía en conversación (h) |
| `three_g` | Soporta 3G (0/1) |
| `touch_screen` | Pantalla táctil (0/1) |
| `wifi` | Tiene wifi (0/1) |
| `price_range` | Categoría de precio: 0=bajo, 1=medio, 2=alto, 3=muy alto |
| `price` | Precio en euros (variable continua) |

No contiene valores nulos y todas las columnas son numéricas, por lo que no
requiere limpieza adicional antes de aplicar los modelos.

## Qué hace el script, paso a paso

`ejerciciofinal.py` se ejecuta de principio a fin y va imprimiendo resultados
por consola, además de guardar todas las gráficas en la carpeta `figuras/`
(en vez de abrir ventanas emergentes, para que funcione igual en local que en
un entorno sin pantalla; si lo ejecutas en tu propio ordenador puedes cambiar
`guardar_figura(...)` por `plt.show()` si prefieres verlas en ventanas).

1. **Carga y exploración**: lee el CSV, muestra las primeras filas y el
   número de registros/columnas.
2. **Correlaciones con `price_range`**: calcula la matriz de correlaciones
   completa y extrae las 5 variables más correlacionadas con el rango de
   precio. `ram` y `price` destacan muy por encima del resto.
3. **Matriz de correlaciones** entre `price`, `ram` y `battery_power`,
   representada como mapa de calor.
4. **Regresión lineal simple** de `price` frente a `ram`: calcula la
   pendiente, el intercepto y el R², dibuja la recta de regresión sobre los
   datos y valora si el ajuste es bueno.
5. **Predicción puntual**: estima el precio de un móvil con 3.100 MB de RAM
   usando el modelo anterior.
6. **Análisis de residuos**: gráfica de residuos frente a valores predichos,
   para detectar patrones que indiquen que el modelo lineal se queda corto.
7. **Clasificación con SVM (kernel lineal)** usando `ram` y `battery_power`
   para predecir `price_range`, con su gráfica de regiones de decisión y la
   exactitud sobre el conjunto de test.
8. **Clasificación con SVM (kernel RBF, gamma=20)**: mismo problema, kernel
   distinto, para comparar y mostrar el efecto de un gamma demasiado alto
   (sobreajuste).
9. **Estrategia One-vs-Rest explícita** (`OneVsRestClassifier`), comparando
   su exactitud con la de los apartados anteriores.
10. **Clustering K-medias** sobre `ram` y `price` sin usar las etiquetas:
    método del codo para elegir el número de clústeres, y comparación visual
    entre `k` y `k+1`.
11. **Clustering DBSCAN y HDBSCAN**: gráfico de distancia al vecino más
    cercano para elegir epsilon, resultado de DBSCAN, y resultado de HDBSCAN
    si el paquete está instalado (es opcional, ver más abajo).
12. **Clustering aglomerativo (jerárquico)** con 4 clústeres.
13. **PCA sobre el dataset completo**: varianza explicada acumulada y número
    de dimensiones necesarias para conservar el 95% de la varianza.

## Resultados principales

- Las variables más relacionadas con el precio son, con diferencia, **`ram`**
  y, lógicamente, **`price`** consigo mismo en `price_range`; el resto de
  variables (batería, resolución de pantalla...) tienen una correlación mucho
  más débil.
- La regresión lineal `price ~ ram` obtiene un **R² ≈ 0,65**: la RAM explica
  buena parte del precio, pero el ajuste es moderado, no excelente — hay
  margen para mejorar añadiendo más variables (un modelo multivariable
  probablemente lo superaría).
- La clasificación de `price_range` a partir de `ram` y `battery_power` con
  **SVM de kernel lineal** alcanza una exactitud en torno al **85%**, lo cual
  indica que estas dos variables (sobre todo `ram`) separan bastante bien las
  cuatro categorías de precio.
- Un kernel **RBF con gamma muy alto (20)** genera fronteras de decisión
  mucho más complejas y ajustadas a los datos de entrenamiento, lo que es un
  ejemplo claro de sobreajuste: hay que tener cuidado al elegir
  hiperparámetros del kernel.
- En el agrupamiento no supervisado, el **método del codo** sugiere unos 3-4
  clústeres como número razonable para `ram` y `price`, coherente con las 4
  categorías reales de `price_range` que existen en los datos.
- El **PCA** muestra que la varianza está bastante repartida entre las 21
  variables (muchas son binarias e independientes entre sí: bluetooth, wifi,
  4G...), por lo que se necesitan bastantes componentes (en torno a 18) para
  retener el 95% de la varianza. Esto es razonable: a diferencia de datasets
  con variables muy redundantes, aquí cada característica aporta información
  relativamente propia.

## Cómo ejecutarlo

### Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

El script funciona perfectamente solo con estas librerías. Adicionalmente,
si quieres ejecutar también el apartado de HDBSCAN, instala:

```bash
pip install hdbscan
```

Si no lo instalas, el script lo detecta automáticamente y omite ese
apartado en concreto sin afectar al resto (mostrará un aviso por consola).

De forma opcional, si quieres usar `mlxtend.plotting.plot_decision_regions`
para las regiones de decisión del SVM en lugar de la función equivalente
incluida en el propio script, puedes instalar:

```bash
pip install mlxtend
```

### Ejecución

Asegúrate de tener el archivo `datos_moviles (2).csv` en la misma carpeta
que el script, y ejecuta:

```bash
python ejerciciofinal.py
```

Verás la salida de cada apartado por consola y, al terminar, todas las
gráficas estarán guardadas en la carpeta `figuras/`.

## Notas sobre las correcciones respecto a la versión anterior

Este script ha sido revisado y corregido respecto a una versión previa que
presentaba varios problemas:

- **Bloques de correlación duplicados**: el cálculo de correlaciones y el
  mapa de calor estaban repetidos por error; se ha dejado una única versión
  consolidada.
- **Faltaban apartados completos** del enunciado: la gráfica de la recta de
  regresión con coeficientes y R² (punto 6), la clasificación SVM con kernel
  RBF y gamma=20 (punto 10), la estrategia OvR explícita (punto 11) y el
  clustering K-medias con el método del codo (punto 12) no estaban
  implementados y se han añadido.
- **Los resultados de clustering no se representaban gráficamente**: DBSCAN,
  HDBSCAN y el clustering aglomerativo se calculaban pero nunca se
  visualizaban ni se imprimían sus resultados; ahora cada método muestra su
  número de clústeres y su gráfica correspondiente.
- **Problema de convergencia en el SVM**: al entrenar el SVM con `ram` y
  `battery_power` sin escalar (variables en rangos muy distintos: cientos
  frente a miles), el solver del kernel lineal no convergía y el script podía
  quedarse colgado. Se ha añadido un escalado (`StandardScaler`) antes de
  entrenar los modelos SVM, tal y como recomienda el propio aviso de
  scikit-learn.
- **DBSCAN con epsilon mal calibrado**: el valor de `eps=0.3` del script
  original se aplicaba sobre datos sin escalar, lo que lo hacía
  prácticamente inútil dada la enorme diferencia de escala entre `ram` (cientos-
  miles) y `price` (decenas-miles). Ahora se estandarizan los datos antes de
  calcular la gráfica de distancias y aplicar DBSCAN, lo que permite elegir
  un epsilon con sentido.
- **`import hdbscan` rompía el script entero** si el paquete no estaba
  instalado, aunque solo se usaba al final. Ahora la importación es opcional
  y el resto del script funciona igualmente sin ese paquete.
- **Bug de tipos al graficar la recta de regresión** (`'numpy.ndarray' object
  has no attribute 'values'`), corregido.
- **`plt.show()` sustituido por guardado en archivo**, para que el script
  pueda ejecutarse igual en local (con ventanas, si se prefiere) que en
  entornos sin interfaz gráfica.

## Autor

Daniel Iturralde — Examen final de Machine Learning, CEINPRO.
