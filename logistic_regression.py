import pandas as pd
import urllib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from operator import itemgetter
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

class logisticRegressionModel(object):
    def __init__(self, folds = 5):

        self.foldNum = folds
        self.folds = ['fold1','fold2', 'fold3', 'fold4', 'fold5']
        # Results
        self.val = []
        self.test = []
        self.val_avg = 0
        self.test_avg = 0 
        self.estimates = object()

        # Data loader
        self.data = pd.read_pickle('data.pkl')

    '''
    Finds the logistic regression of each fold and
    returns the accuracies along with the average
    '''
    def run_regression(self):
        for i in self.folds:
            fold = self.data[i]
            fold_train = fold['train']
            fold_val = fold['val']
            fold_test = fold['test']
            xtrain, ytrain = fold_train['x'],fold_train['y']
            xval, yval = fold_val['x'], fold_val['y']
            xtest, ytest = fold_test['x'],fold_test['y']
            self.val.append(self.logistic_regression(xtrain, ytrain, xval, yval ))
            self.test.append(self.logistic_regression(xtrain, ytrain, xtest, ytest ))
        
        self.val_avg = np.average(self.val)
        self.test_avg = np.average(self.test)
  
    '''
    Uses the sklearn Logistic Regression method to
    fit the model to the training data and returns
    the scores
    '''
    def logistic_regression(self, xtrain, ytrain, x, y):
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(xtrain.astype('int') , ytrain.astype('int') )
        clf.predict(x.astype('int'))
        self.graph_results(xtrain, clf.predict_proba(xtrain))
        return clf.score(xtrain.astype('int') , ytrain.astype('int') )

    def print_results(self):
        print('______________________________________________________')
        print('          |                   ACCURACY                ')
        print('     FOLD |        VAL            |        TEST       ')
        print('----------+-----------------------+-------------------')
        
        for i in range(0,self.foldNum):
            print ('  ',i +1 ,'     |  ' ,self.val[i],'  |  ',self.test[i])

        print('______________________________________________________')
        print('AVG       |  ', self.val_avg, ' |  ', self.test_avg)
        print('______________________________________________________')

    def graph_results(self, x, xLR):
        plt.plot(x)
        plt.plot(xLR)
        plt.ylabel("Probability of Skin")
        plt.show()


if __name__ == '__main__':
    model = logisticRegressionModel()
    model.run_regression()
    model.print_results()
