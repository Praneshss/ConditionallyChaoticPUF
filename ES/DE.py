#
# Implemented by Saranyu Chattopadhyay
#
import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

def hamming(a, b):
    return len([i for i in filter(lambda x: x[0] != x[1], zip(a, b))])

def fobj(x, X_valid, Y_valid):
        (m, n) = X_valid.shape

        Y_pred = []

        for i in range(m):
                temp = np.dot(x, X_valid[i][:])
                if temp>0:
                        Y_pred.append(1)
                else:
                        Y_pred.append(0)

        acc = hamming(Y_valid, Y_pred)
        #print('Done.\nAccuracy: %f' % acc)
        return (acc/len(Y_valid))



def de(fobj, bounds, X_valid, Y_valid, mut=0.8, crossp=0.7, popsize=20, its=1000):
        dimensions = len(bounds)
        pop = np.random.rand(popsize, dimensions)
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([fobj(ind, X_valid, Y_valid) for ind in pop_denorm])
        #print(fitness)
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        for i in range(its):
                print(i)
                for j in range(popsize):
                        idxs = [idx for idx in range(popsize) if idx != j]
                        a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
                        mutant = np.clip(a + mut * (b - c), 0, 1)
                        cross_points = np.random.rand(dimensions) < crossp
                        if not np.any(cross_points):
                                cross_points[np.random.randint(0, dimensions)] = True
                        trial = np.where(cross_points, mutant, pop[j])
                        trial_denorm = min_b + trial * diff
                        f = fobj(trial_denorm, X_valid, Y_valid)
                        if f < fitness[j]:
                                fitness[j] = f
                                pop[j] = trial
                                if f < fitness[best_idx]:
                                        best_idx = j
                                        best = trial_denorm
        return best, fitness[best_idx]


 X = np.array(list(csv.reader(open("challenge_50000.csv", "r"), delimiter=","))).astype(np.float32)


for i in [0,0.25,0.5,0.75,1]:
        Y = []
        with open("response_mixed_50000_"+str(i)+".csv", "r") as f:
                reader = csv.reader(f)
                for row in reader:
                        Y.append(int(row[0]))
                Y = np.array(Y)
                
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        clf = LogisticRegression()
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print('LR Done.\nAccuracy: %f\n' % accuracy_score(Y_test, Y_pred))
        coef = clf.coef_
        bounds= [(i*0.8, i*1.2) for i in coef[0][:]]
        coef, best  = de(fobj, bounds, X_train, Y_train)
        (m, n) = X_test.shape
        Y_pred = []

        for i in range(m):
                temp = np.dot(coef, X_test[i][:])
                if temp>0:
                        Y_pred.append(1)
                else:
                        Y_pred.append(0)


        print('DE Done.\nAccuracy: %f\n' % accuracy_score(Y_test, Y_pred))

       
