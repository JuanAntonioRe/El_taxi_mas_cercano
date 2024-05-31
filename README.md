# El taxi más cercano
Con este proyecto se hace uso de módulos que tienen como base el álgebra lineal de la librería `sklearn`.

Esto quiere decir que los modelos hacen uso de operaciones matriciales para entrenarse y poder realizar una predicción. Tales operaciones engloban el producto escalar para encontrar la distancia Euclidiana o la distancia Manhattan. Ambas muy útiles para el entrenamiento de modelos de Machine Learning.

## Descripción del proyecto
Una compañía de seguros llamada Sure Tomorrow quiere resolver las siguientes tareas:
1. Encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.
2. Predecir si es probable que un nuevo cliente reciba un beneficio de seguro. Y ¿puede un modelo de predicción funcionar mejor que un modelo ficticio?
3. Predecir la cantidad de beneficios de seguro que probablemente recibirá un nuevo cliente utilizando un modelo de regresión lineal.
4. Proteger los datos personales de los clientes sin romper el modelo de la tarea anterior.

## Objetivo
 Desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento de datos u ofuscación de datos. Pero los datos deben protegerse de tal manera que la calidad de los modelos de machine learning no se vea afectada.

 ## Desarrollo del proyecto
### Preprocesamiento y exploración de datos
Se importan los datos y se checan las estadísticas descriptivas de los datos.

También se realiza un análisis exploratorio de datos para tratar de detectar grupos obvios  en los clientes, sin embargo no es posible sin la ayuda del Machine Learning.

### Tarea 1
Se desarrolla una función que devuelva los k vecinos más cercanos para un $n^{th}$ objeto basándose en una métrica de distancia especificada. A la hora de realizar esta tarea no debe tenerse en cuenta el número de prestaciones de seguro recibidas.

Se hacen 4 casos:
- los datos no están escalados
- los datos se escalan con el escalador [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
- Métricas de distancia
  - Euclidiana
  - Manhattan

Respondiendo a las preguntas: ¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?- ¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?

### Tarea 2
Se construye un clasificador basado en KNN y mide su calidad con la métrica F1 para k=1...10 tanto para los datos originales como para los escalados. Al hacer esto, se observa cómo k puede influir en la métrica de evaluación y si el escalado de los datos provoca alguna diferencia.

Para esto los datos, correspondientes a las etapas de entrenamiento/prueba, se dividen con una proporción 70:30.

### Tarea 3
Se construye una implementación propia de regresión lineal. Y se comprueba la RECM tanto para los datos originales como para los escalados. 

Con esto se puede responder: ¿Hay alguna diferencia en la RECM con respecto a estos dos casos?

### Tarea 4
Se ofuscan los datosa al multiplicar las características numéricas (recuerda que se pueden ver como la matriz $X$) por una matriz invertible $P$. 

$$
X' = X \times P
$$

Después se comprueba cómo quedarán los valores de las características después de la transformación. Para esto la propiedad de invertibilidad es importante aquí, así que $P$ debe ser realmente invertible.

## Conclusiones
Se logran realizar con éxito todas las tareas.

## Tecnologías
* Pandas
* Numpy
* Seaborn
* Math
* sklearn: 
    + train_test_split
    + LinearRegression
    + neighbors.NearestNeighbors
    + neighbors.NeighborsClassifier
    + preprocessing
    + utils.shuffle
    + mean_squared_error
    + StandardScaler