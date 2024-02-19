EXAMEN FINAL DE MACHINE LEARNING DE CEINPRO 



Imagina que trabajas en el departamento de marketing de una importante compañía de telefonía móvil. En los últimos meses habéis procedido a la obtención de numerosos datos relativos a diferentes modelos de teléfonos, estableciendo los precios de venta como variable principal a analizar. El planteamiento genérico del estudio que os proponéis realizar consiste en tratar de fijar un precio razonable, teniendo en cuenta el mercado, para los futuros modelos de móviles que vais a comercializar, en función de variables como el tamaño de memoria RAM, el de memoria interna, píxeles de resolución, etc.

En los siguientes apartados te iré planteando una serie de cuestiones encaminadas a desarrollar este estudio aplicando diversas técnicas de Machine Learning.

1. El dataset que emplearemos en este caso, al tratarse de un ejercicio de naturaleza formativa, procede de kaggle. ¿Podrías indicar de qué tipo de plataforma estamos hablando?

2. El dataset al que hacemos referencia se denomina datos_moviles.csv y puedes descargarlo al final de la descripción de este proyecto. ¿Qué tipo de archivo es este? ¿Cómo se puede cargar esta tipología de ficheros mediante la biblioteca estándar de Python?

3. Carga el fichero datos_moviles.csv en Python, de modo que los datos se guarden en un DataFrame, y escribe el código necesario para visualizar las primeras filas del dataset y para saber cuántos registros, así como el número de características que tiene.

4. En primer lugar, nos centraremos en la variable etiqueta price_range o rango de precios. Esta variable toma el valor 0 para un coste bajo del móvil, 1 para coste medio, 2 para coste alto y 3 para coste muy alto (por ejemplo, un móvil de lujo). Determina las correlaciones existentes entre todas las variables, pero, en particular, céntrate en las relaciones con price_range. ¿Cuáles son las 5 variables que tienen mayor correlación con price_range?

5. Dado que price (el precio en euros de cada móvil) es una variable continua, más interesante para nuestra investigación que range_price, procede a representar gráficamente la matriz de correlaciones considerando las dos variables más correlacionadas con price (excluyendo a price_range, que sirve para etiquetar los móviles en función de dicho precio): ram y battery_power. Recuerda incluir en la matriz a la propia variable price.

6. Procede a obtener la regresión lineal de la variable price frente a la variable ram. Genera la representación gráfica, determina los coeficientes de regresión y los de determinación. ¿Se alcanza un buen ajuste?

7. Si quisieras fijar el precio de un móvil con 3100 MB de memoria RAM, considerando el anterior ajuste lineal, ¿qué valor establecerías?

8. Representa gráficamente los residuos obtenidos frente a los valores predichos según el modelo de regresión lineal generado (ten en cuenta que los precios de los móviles oscilan aproximadamente entre 20 y 2000 €).

9. Céntrate a continuación en las variables ram y battery_power, considerando price_range como una etiqueta de clasificación. Genera una clasificación del conjunto mediante un kernel lineal, incorporando, si puedes, la función plot_decisions_regions para mejorar la salida gráfica. Determina también la exactitud del test. Nota: carga los datos de price_range mediante la instrucción price_range=np.array(data['price_range']), a fin de que no tengas problemas con la dimensión de los arrays (recuerda transponerlo seguidamente).

10. ¿Qué resultado obtendrías si aplicas una clasificación, en el caso anterior, de base radial gaussiana con gamma = 20?

11. Aplica una estrategia OvR para realizar la clasificación de los datos (el Foro de trabajo 2 te proporcionará pautas para ello) y determina de nuevo la exactitud del algoritmo.

12. Supón ahora que no dispones del etiquetado de datos (es decir, de la variable price_range). Considerando las variables ram y price trata de obtener los posibles agrupamientos del conjunto de todos los datos mediante el algoritmo de k-medias. ¿Qué número de clústeres deberías plantear? Obtén la solución para un número de clústeres superior en una unidad. Compara los dos resultados observando las correspondientes gráficas.

13. Obtén ahora los agrupamientos mediante el método DBSCAN y, si re resulta posible, con el método HDBSCAN. Recuerda que en el Foro de trabajo 3 ya has tratado sobre ambos métodos. Para el método DBSCAN investiga un posible valor de épsilon que proporcione un agrupamiento que te resulte razonable y para HDBSCAN emplea el recomendado por las personas que lo han desarrollado.

14. Aplica el algoritmo de agrupamiento por aglomeración al conjunto de datos, considerando el número que consideres más adecuado de clústeres.

15. Considera ahora el dataset completo. Aplica el algoritmo PCA y obtén y representa la varianza explicada en función del número de dimensiones. ¿Cuántas dimensiones requerirás para salvaguardar una varianza en torno al 95 %?
