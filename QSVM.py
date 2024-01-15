from sklearn.svm import SVC
import numpy as np
class QSVM:

    def __init__(self, feature_map, train_features, train_label):
        def kernel(x1, x2):
          
            ψ1 = feature_map(x1.T)
            ψ2 = feature_map(x2.T)
            return np.array([[abs(a.conjugate().T.dot(b))**2 for a in ψ2] for b in ψ1])
        self.svc = SVC(kernel=kernel)
        self.svc.fit(train_features, train_label)

    def run(self):
        pass

    def predict(self, dataset_features):
        return self.svc.predict(dataset_features)
    
    
class noisy_QSVM:

    def __init__(self, feature_map, train_features, train_label,noise):
        def inner(a,b):
            qc = a.compose(b.inverse())
            qc.save_density_matrix()
            sim = AerSimulator(method='density_matrix', noise_model = noise_model)
            return sim.run(qc).result().data()['density_matrix'].data[0][0]
        
        def kernel(x1, x2):
            
            ψ1 = feature_map(x1.T)
            ψ2 = feature_map(x2.T)
            return np.array([[abs(a.conjugate().T.dot(b))**2 for a in ψ2] for b in ψ1])
        self.svc = SVC(kernel=kernel)
        self.svc.fit(train_features, train_label)

    def run(self):
        pass

    def predict(self, dataset_features):
        return self.svc.predict(dataset_features)
