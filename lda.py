"""
SCRIPT AUXILIAR QUE CONTIENE EL MODELO DE MACHINE LEARNING DE LDA - LINEAR DISCRIMINANT ANALYSIS
"""

import numpy as np

class LinearDiscriminant:
    """
    Linear Discriminant Analysis (LDA) class.

    This class implements the LDA algorithm for dimensionality reduction and 
    finding the linear combination of features that best separates two or more 
    classes of objects or events.

    Attributes:
        weights (np.ndarray): The linear discriminants (eigenvectors) that can
            be used to transform the data into a lower-dimensional space.
    """

    def __init__(self, n_components=None):
        """
        Initializes the LinearDiscriminant instance with the number of components.
        
        Args:
            n_components (int, optional): Number of linear discriminants to retain. If None,
                                        all components are kept.
        """
        self.n_components = n_components
        self.weights = None
        self.class_means = None
        self.priors = None
        self.class_labels = None

    def fit(self, X, y):
        """
        Fit the LDA model according to the given training data.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features), 
                where n_samples is the number of samples and n_features is 
                the number of features.
            y (np.ndarray): Target values, shape (n_samples,), where n_samples
                is the number of samples.

        Returns:
            None
        """
        # MEDIDAS PREVENTIVAS PORQUE A VECES RECONOCIA X COMO DATAFRAME EN VEZ DE MATRIZ
        X = np.asarray(X)
        y = np.asarray(y)

        n_features = X.shape[1]
        self.class_labels = np.unique(y)
        n_classes = len(self.class_labels)

        # OBTENEMOS N_COMPONENTS EN CASO DE NO HABERLO COMPLETADO AL LLAMAR AL MODELO
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)

        # PARA CALCULAR LA VARIANZA INTRA CLASE NECESITAMOS CALCULAR LA MEDIA DE CADA CLASE
        mean_vectors = [np.mean(X[y == cls], axis=0) for cls in self.class_labels] # dim de cada vector (3, 37)

        # PARA CALCULAR LA DISPERSION ENTRE CLASES NECESITAMOS LA MEDIA GLOBAL
        overall_mean = np.mean(X, axis=0)

        Sw = np.zeros((n_features, n_features))
        Sc = np.zeros((n_features, n_features))

        for i, mean in enumerate(mean_vectors):
            # VARIANZA INTRA CLASE 
            Xc = np.array(X[y==self.class_labels[i]])
            class_scatter = (Xc - mean).T @ (Xc - mean)
            Sw += class_scatter

            # DISPERSION ENTRE CLASES ES NUM_OBSERVACIONES DE CADA CLASE POR LA VARIANZA GLOBAL
            Nc = Xc.shape[0]
            mean_diff = (overall_mean - mean).reshape(n_features, 1)
            Sc += Nc * (mean_diff @ mean_diff.T)
        
        # AHORA LO QUE BUSCAMOS ES MAXIMIZAR LA DISPERSION ENTRE CLASES A LA VEZ QUE MINIMIZAR LA VARIANZA INTRACLASE
        # PARA ELLO HACEMOS Sw^-1 * Sc Y OBTENEMOS LOS AUTOVALORES Y AUTOVECTORES QUE INDICAN AL MODELO LA PROYECCION OPTIMA DE LOS DATOS

        eigen_values, eigen_vectors = np.linalg.eig(Sc @ np.linalg.pinv(Sw)) # dim de los eigen_vectors (num_eigen_vectors, 37)

        # BUSCAMOS OBTENER LOS PRIMEROS N_COMPONENTS AUTOVECTORES CON MAYOR TAMAÑO (MAYOR AUTOVALOR ASOCIADO)
        eigen_pairs = sorted([(eigen_values[i], eigen_vectors[:,i]) for i in range(len(eigen_values))], key = lambda x: x[0], reverse=True)

        self.weights = np.array([pair[1] for pair in eigen_pairs[:self.n_components]]).T.real #tras trasponer nos queda dim(37, n_comp)

        self.class_means = [mean_vector @ self.weights for mean_vector in mean_vectors] # cada mean_vector es (3,37)@(37,n_comp)= dim(3,n_comp)

        # DETERMINAMOS LAS PROBABILIDADES A PRIORI DE CADA CLASE (MEDIA PONDERADA PARA CADA CLASE)
        self.priors = np.array([np.mean(y==cls) for cls in self.class_labels])

    def transform(self, X):
        """
        Project the data onto the top linear discriminants.
        
        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).
        
        Returns:
            X_transformed (np.ndarray): Data projected onto the selected linear discriminants,
                                        shape (n_samples, n_components).
        """
        X = np.asarray(X)
        X_transformed = X @ self.weights # dim(X) = (n_samples,n_features)@(n_features, n_comps)=(n_samples, n_comps)
        return X_transformed
    

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values, shape (n_samples,).

        Returns:
            X_lda (np.ndarray): Transformed data, shape (n_samples, n_components),
                where n_components <= n_classes - 1.
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        This method applies the learned linear discriminant analysis model to
        predict the class labels of the given samples. The prediction is based
        on the Bayesian discriminant score, which combines the distance to each
        class mean in the transformed space with the prior probability of each class.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features), where
                n_samples is the number of samples and n_features is the number
                of features.

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,), where each
                entry is the predicted class label for the corresponding sample
                in X.
        """
        # REDUCIMOS EL ESPACIO DE SAMPLES A LA DE COMPONENTES PARA PROYECTARLA CON LDA
        X_transformed = self.transform(X)

        # CALCULAMOS EL SCORE BAYESIANO DE CADA PUNTO PARA CADA CLASE
        #   1) -0.5 * DISTANCIA AL CUADRADO A LA MEDIA DE LA CLASE CUANTO MAS CERCA ESTE EL PUNTO A LA MEDIA DE UNA CLASE, MAYOR SERA SU SCORE
        #   2) LOG(PRIOR) DE LA CLASE (TERMINO PROBABILISTICO) CLASES MAS FRECUENTES EN EL ENTRENAMIENTO RECIBEN UN BONUS ADICIONAL

        scores = []
        for i, cls_mean in enumerate(self.class_means):
            # DISTANCIA EUCLIDIANA AL CUADRADO DE CADA PUNTO A LA MEDIA DE LA CLASE i
            dist = np.sum((X_transformed - cls_mean) ** 2, axis=1)
            # SCORE BAYESIANO: PENALIZAMOS LA DISTANCIA Y BONIFICAMOS POR LA FRECUENCIA DE LA CLASE
            score = -0.5 * dist + np.log(self.priors[i])
            scores.append(score)

        # APILAMOS LOS SCORES EN UNA MATRIZ (n_samples, n_classes)
        # CADA FILA ES UN SAMPLE Y CADA COLUMNA ES EL SCORE PARA UNA CLASE
        scores = np.stack(scores, axis=1)

        # ASIGNAMOS A CADA SAMPLE LA CLASE CON MAYOR SCORE BAYESIANO
        return self.class_labels[np.argmax(scores, axis=1)]