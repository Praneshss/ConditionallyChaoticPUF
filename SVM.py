
import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
import pandas as pd

df1=pd.read_csv('final_challenge.csv',delim_whitespace=True,header=None)
X = df1.iloc[:500000,:64].values.astype(np.int32)

df2=pd.read_csv('mixture_25_75.csv',header=None)
Y=df2.iloc[:,:].values.astype(np.int32)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

x_train = X_train[:10000,:]
y_train = Y_train[:10000,:]
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)

Y_pred = svclassifier.predict(X_test)
print('\n Linear SVM Accuracy: %f\n' % accuracy_score(Y_test, Y_pred))


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(x_train, y_train)

Y_pred = svclassifier.predict(X_test)
print('\n RBF SVM Accuracy: %f\n' % accuracy_score(Y_test, Y_pred))
