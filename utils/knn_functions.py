import numpy as np
import pandas as pd
import sklearn.neighbors as skl_nb
import matplotlib.pyplot as plt

# Model Iteration
def model_iterator(X_train, y_train, X_test, y_test, feature_combinations, iterations):
    results_column_names = [
        'number_words_female',
        'total_words',
        'number_of_words_lead',
        'difference_in_words_lead_and_co_lead',
        'number_of_male_actors',
        'year',
        'number_of_female_actors',
        'number_words_male',
        'gross',
        'mean_age_male',
        'mean_age_female',
        'age_lead',
        'age_co_lead',
        'best_k',
        'lowest_misclassification',
        'iteration_no'
    ]

    results = pd.DataFrame(columns=results_column_names)

    for iteration in range(1, iterations + 1):
        best_k, lowest_misclassification = find_best_k_with_misclassification(
            X_train[feature_combinations[iteration]], y_train, X_test[feature_combinations[iteration]], y_test, 30)

        row = {
            'number_words_female': 0,
            'total_words': 0,
            'number_of_words_lead': 0,
            'difference_in_words_lead_and_co_lead': 0,
            'number_of_male_actors': 0,
            'year': 0,
            'number_of_female_actors': 0,
            'number_words_male': 0,
            'gross': 0,
            'mean_age_male': 0,
            'mean_age_female': 0,
            'age_lead': 0,
            'age_co_lead': 0,
            'best_k': best_k,
            'lowest_misclassification': lowest_misclassification,
            'iteration_no': iteration
        }

        for key, value in row.items():
            if key in feature_combinations[iteration]:
                row[key] = 1
            else:
                pass

        results = results.append(row, ignore_index=True)

    return results

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