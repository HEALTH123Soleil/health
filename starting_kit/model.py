
import pickle
import numpy as np   
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.random_projection import sparse_random_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest

class model (BaseEstimator):
    def __init__(self):
      
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.svd = TruncatedSVD(n_components = 10)
        self.classifier= SVC(C=1.)
        
        pipeline = Pipeline([('svd', TruncatedSVD()),('skb',SelectKBest()),( 'clf', SGDClassifier())])
        
        parameters = {
              'pca__n_components': (0,1, 2, 3),
               # 'vect__max_features': (None, 5000, 10000, 50000),
               #'pca__svd_solver': ((1, 1), (1, 2)),  # unigrams or bigrams
               # 'tfidf__use_idf': (True, False),
               # 'tfidf__norm': ('l1', 'l2'),
              'svd__n_components': (0,1,2,3,4,5,6,7),
              'clf__max_iter': (5,),
              'clf__alpha': (0.00001, 0.000001),
              'clf__penalty': ('l2', 'elasticnet'),
}
        
        grid_search = GridSearchCV(pipeline, parameters, cv=5,n_jobs=-1, verbose=1)

    def fit(self, X, y):
       
        self.svd.fit(X)
        X = self.svd.transform(X)
        self.classifier.fit(X,y)
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.is_trained=True
        
        
      
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    def predict(self, X):
       
    function predict eventually can return probabilities.
        
         print("Matrix Factorization of test set by SVD")
            svd = TruncatedSVD(n_components = 500)

            A = svd.fit_transform(X)
            T = svd.components_
        
            print("Shape of A :", A.shape)
            print("Shape of T :", T.shape)
        X = self.svd.transform(X)
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = self.classifier.predict(X)
        # If you uncomment the next line, you get pretty good results for the Iris data :-)
        #y = np.round(X[:,3])
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
