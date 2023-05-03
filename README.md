# SimSiam
SimSiam with ResNet34, VGG13, AlexNet, SqueezeNet1.1 and MobileNetV2

Para realizar los diferentes experimentos se ha adaptado el código original de \textit{SimSiam} creado por Facebook Research https://github.com/facebookresearch/simsiam para cada una de las redes neuronales convolucionales usadas. Se ha desarrollado usando Python y la librería \textit{Pytorch}. Las principales funcionalidades se pueden encontrar en los archivos:

SupervisedClassifier.ipynb: permite entrenar de manera supervisada las redes neuronales convolucionales ResNet34, VGG13, AlexNet, MobileNetV2 y SqueezeNet1.1. Podremos indicar qué hiperparámetros usar como la arquitectura, épocas, base de datos, \textit{learning rate}, \textit{weight decay}, \textit{learning rate scheduler}, \textit{momentum} y el \textit{batch size}. De manera general se usa el optimizador SGD y \textit{Cross-Entropy loss}.

SimSiamTrainer.ipynb: permite entrenar los extractores de características de las redes convolucionales nombradas con SimSiam. Utiliza la clase de SimSiam adaptada. Podremos indicar el batch size, número de épocas, learning rate base, momentum y weight decay. Tambíen realiza como validación una clasificación kNN con k=1.

TSNEFeatures.ipynb: permite proyectar en 2D en el plano mediante el algoritmo t-SNE, implementado en la librería de Python sklearn, las salidas de los encoders preentrenados con SimSiam.

SimSiamClassifier.ipynb: permite realizar el entrenamiento de los clasificadores auto-supervisados. Para pode realizar la clasificación, tendremos que indicar al programa el path del modelo preentrenado con SimSiam. Permite al usuario modificar el batch size, número de épocas, base de datos, learning rate, momentum y weight decay.
