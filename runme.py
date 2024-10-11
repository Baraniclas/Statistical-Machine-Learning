#importing all neccesary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.discriminant_analysis as skl_da
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# loading the data
data = pd.read_csv('/content/training_data.csv', dtype={'ID': str}).dropna().reset_index(drop=True)

data['increase_stock'] = data['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})

# Plotting
data_snow = data[['snowdepth', 'increase_stock']]
plot_snow = sns.pairplot(data_snow, hue = 'increase_stock', height = 6);

data_precip = data[['precip', 'increase_stock']]
plot_precip = sns.pairplot(data_precip, hue = 'increase_stock', height = 6);

#do not modify this seed value, shuffle the train data in an another part of the code, so we all have the same test set

# Categorizing different variables
data['is_winter'] = data['month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
data['is_spring'] = data['month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
data['is_summer'] = data['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
data['is_fall'] = data['month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
data['rush_hour'] = data['hour_of_day'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
data['night_time'] = data['hour_of_day'].apply(lambda x: 1 if (20 <= x) or (7 >= x) else 0)
data['is_there_snow'] = data['snowdepth'].apply(lambda x: 1 if (0 < x) else 0)
data['increase_stock'] = data['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})

# dropping variables from the dataframe
columns_to_drop = ['snowdepth','summertime','day_of_week','month','dew', 'precip', 'snow']
data.drop(columns=columns_to_drop, inplace=True)

#normalizing numerical coloumns
normalized_cols = ['temp', 'humidity', 'windspeed', 'cloudcover', 'visibility']
dummy_cols = ['hour_of_day', 'holiday', 'weekday']

scaler = StandardScaler()
data[normalized_cols] = scaler.fit_transform(data[normalized_cols])

# creating data set with all the data to train the final chosen method
x_final = data.drop(columns = ['increase_stock'])
y_final = data['increase_stock']

seed_value = 42
#Splitting the data into training set and a testing set
data, test_data = skl_ms.train_test_split(data, test_size=0.2, random_state=seed_value, shuffle = False)

# seperating labels and input data
x_training = data.drop(columns = ['increase_stock'])
y_training = data['increase_stock']
x_testing = test_data.drop(columns = ['increase_stock'])
y_testing = test_data['increase_stock']

# assigning models
LDA = skl_da.LinearDiscriminantAnalysis()
QDA = skl_da.QuadraticDiscriminantAnalysis()

# splitting data into training set and validation set
x_train, x_val, y_train, y_val = skl_ms.train_test_split(x_training, y_training, train_size = 0.80, random_state = 1 )

# fit the basic models
LDA.fit(x_train, y_train)
QDA.fit(x_train, y_train)

# accuracy on training data
LDA_score_train = LDA.score(x_train, y_train) # 0.869
QDA_score_train = QDA.score(x_train, y_train) # 0.196

# accuracy on validation data
LDA_score_val = LDA.score(x_val, y_val) # 0.859
QDA_score_val = QDA.score(x_val, y_val) # 0.164

# Tuning LDA, grid search for best solver
# tuning done on the left out validation set
# increasing accuracy to 0.898
param_grid = {'solver': ['svd', 'lsqr', 'eigen']}
grid_search = skl_ms.GridSearchCV(LDA, param_grid, cv=5)
grid_search.fit(x_val, y_val)

best_paramss = grid_search.best_params_

best_LDA = grid_search.best_estimator_

y_pred = best_LDA.predict(x_val)
accuracy = accuracy_score(y_val, y_pred) # LDA tuned result = 0.848

# Tuning QDA, grid search to optimize the regularization parameter
# tuning done on the left out validation set
# increasing accuracy to 0.887
params = [{'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}]
cv = skl_ms.RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
search = skl_ms.GridSearchCV(QDA, params, cv = cv)
search.fit(x_val, y_val)


best_params = search.best_params_
best_QDA = search.best_estimator_

y_pred = best_QDA.predict(x_val)
acc = accuracy_score(y_val, y_pred)

print('Accuracy on Train data')
print(f'LDA: {LDA_score_train:.3f}')
print(f'QDA: {QDA_score_train:.3f}')
print('Val data')
print(f'LDA: {LDA_score_val:.3f}')
print(f'QDA: {QDA_score_val:.3f}')
print('Grid Search alteration')
print(f'Grid Search result on LDA {accuracy:.3f}')
print(f'Grid search result on QDA {acc:.3f}');

#train the tuned models
best_LDA.fit(x_training, y_training)
best_QDA.fit(x_training, y_training)

accuraccy_rates=[]
# Specified parameters
params = {
    'C': 1.0,
    'class_weight': None,
    'fit_intercept': True,
    'intercept_scaling': 1,
    'max_iter': 100,
    'penalty': 'l1',
    'solver': 'liblinear',
    'tol': 0.001
}

#Cross validation

n_fold=10
cv=skl_ms.KFold(n_splits=n_fold, random_state=2, shuffle=True)

for train_index, val_index in cv.split(x_training):
  X_train, X_val = x_training.iloc[train_index], x_training.iloc[val_index]
  y_train, y_val = y_training.iloc[train_index], y_training.iloc[val_index]

  model=skl_lm.LogisticRegression(**params)
  model.fit(X_train, y_train)
  prediction = model.predict(X_val)
  accuracy=np.mean(prediction==y_val)
  accuraccy_rates.append(accuracy)
  print('accuracy rate: ' + str(accuracy))

print('Mean accuracy rate: ' + str(np.mean(accuraccy_rates)))

# Define hyperparameters grid
param_grid = {
    'C': [0.1, 1.0, 10.0, 30.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],  # Different solvers
    'max_iter': [100, 200, 300],  # Different max_iter values
    'class_weight': [None, 'balanced'],  # Different class weight options
    'fit_intercept': [True, False],  # Whether to fit intercept
    'tol': [1e-3, 1e-4],  # Tolerance for stopping criteria
    'intercept_scaling': [1, 2]  # Scaling for intercept (relevant for 'liblinear')
    }

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=skl_lm.LogisticRegression(solver='liblinear'),
                          param_grid=param_grid,
                           cv=cv)

# Fit the model
grid_search.fit(X_val, y_val)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best model
best_LG = grid_search.best_estimator_
# Fit the best LG model
best_LG.fit(x_training, y_training)
## Split the data into training(80%) and validation sets(20%)
x_train, x_val, y_train, y_val = skl_ms.train_test_split(x_training, y_training, test_size=0.2, random_state=1)

# Implement and evaluate k-NN Classifier with k=2
knn = skl_nb.KNeighborsClassifier(n_neighbors=2).fit(x_train, y_train)
print("Test value: k = 2")
print(f"Accuracy: {accuracy_score(y_val, knn.predict(x_val))}\n")

# Experiment with different values of k to find the optimal one
misclassification = []
k_values = range(1, 50)  # 50 - 200 is optimal

for k in k_values:
    knn = skl_nb.KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_val)
    misclassification.append(np.mean(y_pred != y_val))

# Plotting misclassification rate vs. k
plt.plot(k_values, misclassification, marker='.')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Misclassification Rate')
plt.title('k-NN Varying number of Neighbors')
plt.show()

# Identify the optimal k value and retrain the model
optimal_k = k_values[misclassification.index(min(misclassification))]
print("For loop")
print(f"Optimal value of k: {optimal_k}")

# Retraining k-NN classifier with the optimal number of neighbors
knn_optimal = skl_nb.KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(x_train, y_train)
y_pred_optimal = knn_optimal.predict(x_val)
accuracy_optimal = accuracy_score(y_val, y_pred_optimal)
print(f"Optimized Accuracy with k = {optimal_k}: {accuracy_optimal}\n")

# Find the optimal k using GridSearchCV
grid_search = GridSearchCV(skl_nb.KNeighborsClassifier(), {'n_neighbors': range(1, 50)}, cv=5, scoring='accuracy').fit(x_train, y_train)
print("Grid search")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best model accuracy: {accuracy_score(y_val, grid_search.best_estimator_.predict(x_val))}\n")

# Find the optimal k using RandomizedSearchCV, 1 - 200 seems to be getting best result
random_search = skl_ms.RandomizedSearchCV(skl_nb.KNeighborsClassifier(), {'n_neighbors': range(1, 200)}, cv=5, scoring='accuracy', random_state=1).fit(x_train, y_train)
print("Random search")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best model accuracy: {accuracy_score(y_val, random_search.best_estimator_.predict(x_val))}")
best_kNN = grid_search.best_estimator_

#training the best kNN model
best_kNN.fit(x_training, y_training)

# seperating the dataset into a training set and a validation set
x_train, x_val, y_train, y_val = skl_ms.train_test_split(x_training, y_training, train_size = 0.80, random_state = 1 )

#fitting the basic model
random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)

# accuracy on training data
random_forest_score_train = random_forest.score(x_train, y_train) # 0.869
random_forest_score_val = random_forest.score(x_val, y_val)

print('Without tuning')
print(f'Random forest accuracy {random_forest_score_train} on training data')
print(f'Random forest accuracy {random_forest_score_val} on validation data')

# Hyperparameters to evaluate over (tuning)
param_grid = { 
    'n_estimators': [25, 50, 100, 150], 
    'max_features': ['sqrt', 'log2', None], 
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9], 
} 

grid_search = GridSearchCV(RandomForestClassifier(), 
                           param_grid=param_grid) 
grid_search.fit(x_val, y_val) 
print(grid_search.best_estimator_)

# Training the best Random Forest model
best_RF = RandomForestClassifier(max_depth = 9, max_features = None, max_leaf_nodes = 6, n_estimators = 25)
best_RF.fit(x_training, y_training)
y_pred = best_RF.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print(f' Accuracy of tuned random forest classifier {accuracy}')

# A dummy classifier for comparison of models.
model = DummyClassifier(strategy = 'uniform')
model.fit(x_training, y_training)
Dummy = model

#final test on hold out dataset

models = []
models.append(Dummy)
models.append(best_LG)
models.append(best_LDA)
models.append(best_QDA)
models.append(best_kNN)
models.append(best_RF)
test_accuracy = []
test_fscore = []

for m in range(np.shape(models)[0]):
        model = models[m]
        pred = model.predict(x_testing)
        test_accuracy.append(np.mean(pred == y_testing))
        test_fscore.append(f1_score(y_testing, pred))


# Plotting test accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(('Dummy', 'LG', 'Lda', 'Qda', 'K-nn', 'Random Forest'), test_accuracy, marker='o', linestyle='-', color='skyblue')
plt.xlabel('Models')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Different Models')
plt.xticks(rotation=45)
plt.grid(True)

# Plotting F1 score
plt.subplot(1, 2, 2)
plt.plot(('Dummy', 'LG', 'Lda', 'Qda', 'K-nn', 'Random Forest'), test_fscore, marker='o', linestyle='-', color='salmon')
plt.xlabel('Models')
plt.ylabel('Test F1 Score')
plt.title('Test F1 Score of Different Models')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

print(f'Mean accuracy per model: {test_accuracy}')
print(f'Mean F score per model: {test_fscore}')

# to be able to download results
from google.colab import files


# training chosen model on the whole training dataset
best_RF.fit(x_final, y_final)

# loading the data
data = pd.read_csv('/content/test_data.csv', dtype={'ID': str}).dropna().reset_index(drop=True)

#do not modify this seed value, shuffle the train data in an another part of the code, so we all have the same test set

# Categorizing different variables
data['is_winter'] = data['month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
data['is_spring'] = data['month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
data['is_summer'] = data['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
data['is_fall'] = data['month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
data['rush_hour'] = data['hour_of_day'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
data['night_time'] = data['hour_of_day'].apply(lambda x: 1 if (20 <= x) or (7 >= x) else 0)
data['is_there_snow'] = data['snowdepth'].apply(lambda x: 1 if (0 < x) else 0)

# dropping variables from the dataframe
columns_to_drop = ['snowdepth','summertime','day_of_week','month','dew', 'precip', 'snow']
data.drop(columns=columns_to_drop, inplace=True)

#normalizing numerical coloumns
normalized_cols = ['temp', 'humidity', 'windspeed', 'cloudcover', 'visibility']
dummy_cols = ['hour_of_day', 'holiday', 'weekday']

scaler = StandardScaler()
data[normalized_cols] = scaler.fit_transform(data[normalized_cols])

y_pred = best_RF.predict(data)
np.savetxt("predictions.csv", y_pred, delimiter=",")

files.download("predictions.csv")