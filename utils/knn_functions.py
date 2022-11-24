import numpy as np
import pandas as pd
import sklearn.neighbors as skl_nb
import matplotlib.pyplot as plt

def find_best_k_with_misclassification(X_train, y_train, X_test, y_test, k_iterations):
    best_k = 0
    lowest_misclassification = 1

    for k in range(1, k_iterations + 1):
        model = skl_nb.KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        misclassification = np.mean(prediction != y_test)
        
        if misclassification < lowest_misclassification:
            best_k = k
            lowest_misclassification = misclassification
        
    return [best_k, lowest_misclassification]

def plot_misclassification(X_train, y_train, X_test, y_test, k_iterations):
    misclassification = []

    for k in range(1, k_iterations + 1): # Try n_neighbors = 1, 2, ...., 50
        model = skl_nb.KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        misclassification.append(np.mean(prediction != y_test))
    
    K = np.linspace(1, k_iterations, k_iterations)
    plt.plot(K, misclassification,'.')
    plt.ylabel('Missclasification')
    plt.xlabel('Number of neighbors')
    plt.show()