
# Factors that influence the salary (wages, earning)
# Exam Example

"""
We are given one-hot encoded panel data on earnings of 595 individuals for the years 1976–1982, originating from the [Panel Study of Income Dynamics](https://rdrr.io/cran/AER/man/PSID7682.html). The data were originally analyzed by Cornwell and Rupert (1988) and employed for assessing various instrumental-variable estimators for panel models.

**Your task is to predict the earnings class (`wage_class`) based on the remaining features.**
"""

"""
A data frame containing 7 annual observations on 12 variables for 595 individuals.


| feature | description |
| --------| -------------|
| `experience` | Years of full-time work experience |
| `weeks` | Weeks worked |
| ` education` | Years of education. |
| `occupation_white` | factor. Is the individual a white-collar ("white"=`True`) or blue-collar ("blue"=`False`) worker? |
| `industry` | factor. Does the individual work in a manufacturing industry? |
| `south_yes` |factor. Does the individual reside in the South? |
| `smsa_yes` |factor. Does the individual reside in a SMSA (standard metropolitan statistical area)? |
| `married_yes` |factor. Is the individual married? |
| `gender_male` | factor indicating a male gender. |
| `union_yes` | factor. Is the individual's wage set by a union contract? |
| `ethnicity_other` |factor indicating ethnicity. Is the individual African-American ("afam") or not ("other")? |
| `wage_class` | **resopnse** $y$: Wage class (`['average, 'high', 'low', 'very high']`) |
"""


# %%
# Basic Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluation & CV Libraries
from sklearn.metrics import precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV


# %%
df_onehot=pd.read_csv('PSID_earnings_onehot.csv', index_col=0)
#df.info()
df_onehot.head()


# %%
#drop missing values
df_onehot.dropna(inplace=True)
df_onehot.isnull().sum()

"""
### (a) Extract the features $X$ and the response (label, target) $y$ of the dataset

- generate a `numpy` array `X` that contains the features $X$.
- generate a `numpy` array `y` that contains the response $y$.
"""
# %%
# START CODE HERE
X = df_onehot.drop(columns=['wage_class'])
y = df_onehot['wage_class']
# END CODE HERE


"""
### (b) Plot a histogram of the response $y$ (`'wage_class'`)
- Are the classes well balanced?
- Answer: ...
"""
# %%
#START CODE HERE
plt.figure(figsize=(15,7))
sns.histplot(data=y)
plt.show()

# Pandas alternative:
# df_onehot['wage_class'].value_counts().plot(kind='bar')

#END CODE HERE


"""
 (c) Split the data in 80% training data and 20% test data
"""
#%%
# Data Pre-processing Libraries
from sklearn.model_selection import train_test_split

# START CODE HERE
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)

# END CODE HERE


"""
(d) Use the `StandardScaler` to standardize the data
"""
#%%
# Data Pre-processing Libraries
from sklearn.preprocessing import StandardScaler

#START CODE HERE
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

#END CODE HERE


"""
### (e) Model Evaluation
the **precision** (macro average: `average='macro'`) on the training and test data for each of these classifiers
Use the following **classifiers as baseline** for your classification and evaluate 

- Random Forest classifier (`RandomForestClassifier`) with standard parameters
- k-nearest neighbors classifier (`KNeighborsClassifier`) with `k=3`
"""
# %%
# Modelling Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

classes=['average', 'high', 'low', 'very high']

#START CODE HERE
random_forest = RandomForestClassifier()
knn = KNeighborsClassifier(3)

random_forest.fit(trainX, trainY)
knn.fit(trainX, trainY)

y_pred_random_forest = random_forest.predict(testX)
y_pred_knn = knn.predict(testX)

accuracy_random_forest = accuracy_score(testY, y_pred_random_forest)
accuracy_knn = accuracy_score(testY, y_pred_knn)
# accuracy_random_forest = random_forest.score(testX, testY)
# accuracy_knn = knn.score(testX, testY)

print(f'Random Forest Accuracy: {accuracy_random_forest:.2f}')
print(f'KNN Accuracy: {accuracy_knn:.2f}')

#END CODE HERE


"""
### (f) Plot a confusion matrix for each classifier and interpret the results

Plot a **Confusion Matrix** for each of the two classifiers, e.g. using
   - `cm = confusion_matrix(y_test, y_pred, labels=model.classes_)`
   - ` disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)`
   

Enter your comments in at least two sentences here:
- ...
- ...
"""
# %%
#START CODE HERE
fig, ax = plt.subplots(2, figsize=(6, 12))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes)
disp.plot(ax=ax[0])
ax[0].set_title("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes)
disp.plot(ax=ax[1])
ax[1].set_title("Knn")

plt.show()

#END CODE HERE


"""
###  (g) Hyperparameter Tuning of random forest

- Tune the hyperparameters a Random Forest Classifier `RandomForestClassifier()` using a 10-fold crossvalidated grid search using `GridSearchCV`. 
- Use the following hyperparameters for your grid search:
    - `params= {'n_estimators':[10,50,100,200], 'max_depth':list(range(1,7))}`
- Use the F1 (`f1_macro`) score as a metric.
- What are the best parameters out of the grid?
"""
# %%
from sklearn.model_selection import GridSearchCV

rf=RandomForestClassifier()
params= {'n_estimators':[10,50,100,200],
         'max_depth':list(range(1,7))}

from sklearn.model_selection import cross_val_score

#START CODE HERE
grid = GridSearchCV(estimator=rf, param_grid=params, scoring='f1_macro', cv=10)
grid.fit(trainX, trainY)

f1_macro_score = grid.score(testX, testY)

print(f'F1 Macro Score: {f1_macro_score:.2f}')
print(f'Best Paramenters {grid.best_params_}')
#END CODE HERE


"""
### (h) Compute and plot the permutation feature importances of the best tuned random forest classifier
- What are the most important factors for a high salary?
"""
#%%
from sklearn.inspection import permutation_importance

#START CODE HERE
# get the best model from the grid search CV
best_model = grid.best_estimator_
# compute the feature importances using permutation test
perm_importances = permutation_importance(
    best_model, testX, testY)
# put them in a Series
forest_importances = pd.Series(perm_importances.importances_mean, index=X.columns)
# sort them (get the indices of the sorted array to be able to apply it on the errors)
sort_index = np.argsort(forest_importances)[::-1]

# plot the importances
fig, ax = plt.subplots()
forest_importances[sort_index].plot.bar(yerr=perm_importances.importances_std[sort_index], ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
#END CODE HERE


"""
###  (i) Hyperparameter Tuning of kNN

- Hypertune a K-nearest neighbour classifier `KNeighborsClassifier()` using a 10-fold crossvalidated grid search. 
- Use the following parameters for your grid search:
    - `params= {'n_neighbors':list(range(20,50))}`
- Use the F1 score (`scoring=f1_macro`) as a metric.
- What are the best parameters for the number of neighbours?
"""
# %%
knn=KNeighborsClassifier()
params= {'n_neighbors':list(range(20,50))}

#START CODE HERE
grid = GridSearchCV(estimator=knn, param_grid=params, scoring='f1_macro', cv=10)
grid.fit(trainX, trainY)

f1_macro_score = grid.score(testX, testY)

print(f'F1 Macro Score: {f1_macro_score:.2f}')
print(f'Best Paramenters {grid.best_params_}')
#END CODE HERE


"""
### (j) Compare and discuss the different approaches

- Considering the classifiers (e/f) where the hyperparameters were not tuned and those where the hyperparameters were tuned (g/i), respectively, which classifier would you recommend and why?

Answers:
- ...
- ... 
"""
# %%
#START CODE HERE


#END CODE HERE



"""
## Upload this notebook as ipynb-File and as html-File (File  →  Download as  →  HTML) to the upload field of this question (2 files are allowed). 
"""

