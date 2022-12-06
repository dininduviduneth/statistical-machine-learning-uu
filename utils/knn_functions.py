import numpy as np
import pandas as pd
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.preprocessing as prep
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Model Iteration
def find_best_k_with_misclassification_cv(X, y, k_iterations, n_fold = 10):

    cv = skl_ms.KFold(n_splits=n_fold, random_state=2, shuffle=True) 
    K = np.arange(1, k_iterations)

    misclassification = np.zeros(len(K))

    for train_index, val_index in cv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        PredictorScaler = StandardScaler()

        # Storing the fit object for later reference
        PredictorScalerFit = PredictorScaler.fit(X_train)
 
        # Generating the standardized values of X and y
        X_train = PredictorScalerFit.transform(X_train)

        # Generating the standardized values of X and y
        X_val = PredictorScalerFit.transform(X_val)
        
        for j, k in enumerate(K):
            model = skl_nb.KNeighborsClassifier(n_neighbors=k) 
            model.fit(X_train, y_train)
            prediction = model.predict(X_val) 
            prediction = prediction.reshape(prediction.shape[0], 1)
            misclassification[j] += np.mean(prediction != y_val)
            
    misclassification /= n_fold

    best_k = K[pd.Series(misclassification).idxmin()]
    lowest_misclassification = min(misclassification)

    '''plt.plot(K, misclassification)
    plt.title('Cross validation error for kNN')
    plt.xlabel('k')
    plt.ylabel('Validation error')
    plt.show()'''

    # print("The K which gives the lowest misclassification error is: " + str(K[pd.Series(misclassification).idxmin()]))
    # print("Lowest misclassification error is: " + str(min(misclassification)))

    return [best_k, lowest_misclassification]

def model_iterator_cv(X, y, feature_combinations, iterations):
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
        if len(feature_combinations[iteration]) >= 7:
            best_k, lowest_misclassification = find_best_k_with_misclassification_cv(
                X[feature_combinations[iteration]], y, k_iterations = 50, n_fold = 10)

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
            results.to_csv(r'/Users/dininduseneviratne/Library/CloudStorage/OneDrive-Uppsalauniversitet/Statistical Machine Learning/project-results/results_8191.csv')
            print(str(iteration) + " OUT OF " + str(iterations) + " ITERATIONS COMPLETED - " + str(iteration*100/iterations) + "%")

        else: 
            pass
    return results

# Function to predict results
def generate_prediction_results(X_train, y_train, X_test, y_test, k_value):
    PredictorScaler = StandardScaler()
 
    # Storing the fit object for later reference
    PredictorScalerFit = PredictorScaler.fit(X_train)
    
    # Generating the standardized values of X and y
    X_train_scaled = PredictorScalerFit.transform(X_train)

    # Generating the standardized values of X and y
    X_test_scaled = PredictorScalerFit.transform(X_test)

    model = skl_nb.KNeighborsClassifier(n_neighbors = k_value)
    model.fit(X_train_scaled, y_train)
    train_prediction = model.predict(X_train_scaled)
    test_prediction = model.predict(X_test_scaled)
    train_misclassification = np.mean(train_prediction != y_train.to_numpy())
    test_misclassification = np.mean(test_prediction != y_test.to_numpy())

    print("Train Misclassification Error: " + str(train_misclassification*100) + "%")
    print("Train Accuracy: " + str((1 - train_misclassification)*100) + "%")
    print("Test Misclassification Error: " + str(test_misclassification*100) + "%")
    print("Test Accuracy: " + str((1 - test_misclassification)*100) + "%")

def generate_prediction_results_without_scaling(X_train, y_train, X_test, y_test, k_value):
    model = skl_nb.KNeighborsClassifier(n_neighbors = k_value)
    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    train_misclassification = np.mean(train_prediction != y_train.to_numpy())
    test_misclassification = np.mean(test_prediction != y_test.to_numpy())

    print("Train Misclassification Error: " + str(train_misclassification*100) + "%")
    print("Train Accuracy: " + str((1 - train_misclassification)*100) + "%")
    print("Test Misclassification Error: " + str(test_misclassification*100) + "%")
    print("Test Accuracy: " + str((1 - test_misclassification)*100) + "%")

# The following functions are used for prior versions - v3 and below
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

# Old normalizer function - not used in v4
def data_normalizer(data_x):
    scaler = prep.MinMaxScaler()
    # d = prep.normalize(data_x, axis=0)
    d = scaler.fit_transform(data_x)
    scaled_data_x = pd.DataFrame(d, columns=data_x.columns)
    
    return scaled_data_x
