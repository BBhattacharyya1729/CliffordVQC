from sklearn.svm import SVC
import numpy as np
class QSVM:

    def __init__(self, feature_map, train_features, train_label,matrix=False):
        def kernel(x1, x2):
            if(matrix == True):
                ψ1 = feature_map(x1)
                ψ2 = feature_map(x2)
                return np.array([[np.trace(a.dot(b)).real for a in ψ2] for b in ψ1])
            ψ1 = feature_map(x1.T)
            ψ2 = feature_map(x2.T)
            return np.array([[abs(a.conjugate().T.dot(b))**2 for a in ψ2] for b in ψ1])
        self.svc = SVC(kernel=kernel)
        self.svc.fit(train_features, train_label)

    def run(self):
        pass

    def predict(self, dataset_features):
        return self.svc.predict(dataset_features)
