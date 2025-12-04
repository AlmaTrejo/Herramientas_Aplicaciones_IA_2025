import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class HyAIA:
    def __init__(self, df):
        self.data = df
        self.columns = df.columns
        self.data_binarios, self.binarios_columns = self.get_binarios()
        self.data_cuantitativos, self.cuantitativos_columns = self.get_cuantitativos()
        self.data_categoricos, self.categoricos_columns = self.get_categoricos()
        self.df_dqr = self.get_dqr()
        
 class LOF:
    """
    Implementación educativa del algoritmo Local Outlier Factor (LOF)
    sin usar sklearn.
    """
    
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        
    def fit(self, X):
        """
        Ajusta el modelo calculando:
        - k-distance
        - vecinos más cercanos
        - densidad local
        - LOF score
        """
        
        # Convertimos a matriz numpy
        self.X = np.array(X)
        
        # Calculamos vecinos
        knn = NearestNeighbors(n_neighbors=self.n_neighbors+1).fit(self.X)
        distances, indices = knn.kneighbors(self.X)
        
        # Quitamos distancia 0 (el mismo punto)
        self.distances = distances[:, 1:]
        self.indices = indices[:, 1:]
        
        # k-distance
        self.k_distance = self.distances[:, -1]
        
        # Reachability distance
        reach_dist = np.maximum(
            self.distances,
            self.k_distance[self.indices]
        )
        self.reachability = reach_dist
        
        # Local reachability density (LRD)
        self.lrd = 1 / (np.mean(self.reachability, axis=1) + 1e-10)
        
        # LOF score
        self.lof_score = np.zeros(len(self.X))
        for i in range(len(self.X)):
            neighbors = self.indices[i]
            self.lof_score[i] = np.mean(self.lrd[neighbors] / self.lrd[i])
        
        return self
    
    def predict(self, threshold=1.5):
        """
        Regresa etiquetas:
        - 1  → normal
        - -1 → outlier
        """
        return np.where(self.lof_score > threshold, -1, 1)
    
    def fit_predict(self, X, threshold=1.5):
        self.fit(X)
        return self.predict(threshold)



