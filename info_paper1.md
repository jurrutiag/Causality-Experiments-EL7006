# Paper 1

## causal_meta/utils/data_utils.py

En este archivo se definen varias funciones entre las cuales están:

### generate_data_categorical:

Se recibe el número de muestras a generar, la distribución marginal y la condicional. Luego, con la distribución marginal se generan muestras con una multinomial de un experimento (categórica), esta muestra categórica luego se pasa a vector one-hot y para cada muestra se samplea luego de la condicional otra categórica, para obtener la otra muestra. Finalmente se entregan ambas muestras obtenidas.

### clase RandomSplineSCM:

Esta clase define una spline para ser utilizada luego en el representation learning.

## Experimentos bivariate-categorical

En estos experimentos se trata la primera parte del paper, donde las distribuciones son bivariadas y categóricas. Además no se trata el representation learning.

### Models.py

Este archivo define modelos para utilizar en estos experimentos. Se define la clase Model como una superclase de los modelos, donde se puede setear la maximum_likelihood, que sería en este caso tomar las probabilidades p(A) y p(B|A) a partir de los datos empíricos y fijarlas como el ground truth.

### Model1 y Model2

Model1 es subclase de Model y define el modelo correcto (A->B), por lo que crea la función set_ground_truth que lo que hace es fijar la probabilidad marginal P(A) y condicional P(B|A) (fija el atributo data del parámetro de cada distribución). Model2 es el modelo incorrecto (B->A), y fija las probabilidades marginal P(B) y condicional P(A|B). En este último caso, el modelo recine P(A) y P(B|A), por lo que debe primero calcular la conjunta, luego marginaliza para encontrar P(B) y luego divide la conjunta con P(B) para obtener P(A|B). En ambos modelos las probabilidades se representan como log probabilidades, por lo que multiplicaciones pasan a sumas y divisiones a restas. Se utiliza también en ambos modelos distribuciones marginales y condicionales categóricas que se encuentran en causal_meta/modules/categorical.py.

En el caso de la marginal categórica, se tiene un módulo con un parámetro que entrega como salida el softmax de este parámetro al hacer forward. En el caso de la condicional sucede lo mismo, pero ahora el parámetro es de dos dimensiones, por lo que el softmax se hace sobre los elementos de los parámetros correspondientes a lo que se condicionó. Todos estos resultados se entregan en logaritmo.

### StructuralModel

Este es el conjunto de ambos modelos, aca se define el modelo A->B y B->A, y se configuran. Este es subclase de BinaryStructuralModel, el cual tiene el parámetro w que en este caso sería gamma (el parámetro estructural), y computa el regret con la función online_loglikelihood en su forward pass.

### Experimento 1: generalization_same_distribution

En este experimento simplemente se crea el modelo estructural y se entrenan los modelos A->B y B->A simultáneamente sobre una distribución de entrenamiento y se muestra que ambos convergen a la misma loss.

### Experimento 2: transfer_adaptation

En este experimento se crea un modelo estructural y para no tener que entrenar sobre una distribución de entrenamiento, simplemente se fija esta distribución con el método set_ground_truth en los modelos actuales. Luego se crea otra distribución marginal para A y se entrena el modelo anterior con datos sampleados de esta distribución de transferencia.

Esto se hace primero para varias distribuciones reales, y luego en cada una de estas distribuciones reales, se realiza para varias intervenciones.

### Experimento 3: meta_learning

En este experimento se intenta aprender el parámetro estructural. Se crea el modelo y se crean 2 optimizadores, SGD para el modelo y RMSProp para el parámetro estructural. Primero se genera la distribución real y se fija esta como la que predice el modelo, luego se realiza una intervención sobre la marginal de A y se realizan k pasos de gradient descent para que el modelo se adapte a esta distribución de transferencia. En cada uno de estos pasos se fue juntando el loss regret, donde se utilizó el forward del StructuralModel (que calcula el regret) con los parámetros de ese momento.


## Experimentos bivariate-categorical


### Experimento 1: data_generation

En este experimento se genera el random spline a ser utilizado luego para el representation learning.

### Experimento 2: meta_learning

En este experimento se entrena utilizando la forma del apéndice G.3, se entrenan entonces primero los dos modelos A->B y B->A, y finalmente se entrena el parámetro estructural que ahora le llaman alpha.

### Experimento 3: encoder_learning

En este experimento se utilizan nuevamente los modelos Gaussianos del apéndice G.3, pero ahora utilizando en los datos un decoder, por lo que se entrena además un encoder. El decoder tiene ángulo -pi/4.
