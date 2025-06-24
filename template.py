

"""
VISUALIZATION       ==========================================================================================
"""
# %% ### Plot classes / Data ###
import seaborn as sns
import matplotlib.pyplot as plt

# Histgramm
plt.figure(figsize=(15,7))
sns.histplot(data=y)
plt.show()

# Boxplot
plt.figure(figsize=(15,7))
sns.boxplot(data=y)
plt.show()

# Scatterplot
plt.figure(figsize=(15,7))
plt.scatter(x, y)
plt.show()

# %% ### Correlation matrix, correlation coefficient as heatmap ###
import pandas as pd
forest_fire = pd.read_csv("forst_fire.csv")

# Correlation matrix visualized in heat map
sns.heatmap(forest_fire.corr(numeric_only = True).round(decimals=1), annot=True, cmap="bwr")
plt.title('Correlation matrix')
plt.show()

# %%
"""
DATA PREPARATION    ==========================================================================================
"""
# %% ### Test, Train Split ###
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
"""
PREPROCESSING       ==========================================================================================
"""
# %% ### Data Handling ###
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline

# Array Handling
a = np.array([1, 2, 3, 4, 5, 6])
a.reshape(2, 3)
    # array([[1, 2, 3],
    #        [4, 5, 6]])

np.mean(a)
np.std(a)
np.sum(a, axis=0)

# Pandas Dataframes
df = pd.DataFrame({
    'age': [22, 25, 47],
    'income': [50000, 62000, 83000],
    'score': [0.5, 0.6, 0.8]
})
df.head()           # Show first rows
df.describe()       # Summary stats
df.info()           # Data types, memory usage
df['income'].mean() # Column-specific operations

x = df.drop(columns=['score'])
y = df['score']     # or y = df['score'].values

df.loc[df["age"] == 25].income.mean() # Mean income of 25 year olds

# Pipeline
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', LinearRegression())
])
pipe.fit(x_train, y_train)
preds = pipe.predict(x_test)

# Standardization and Transformation
scaler = StandardScaler()
scaler = MinMaxScaler()
scaler = PowerTransformer()
scaler = PolynomialFeatures(degree=2)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

df['log_income'] = np.log(df['income'] + 1) # +1 to avoid log(0)

# Feature selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
importances = rf.feature_importances_
feat_importance = pd.Series(importances, index=x.columns).sort_values(ascending=False) # Combine with feature names
print(feat_importance)

# %%
"""
LEARNERS            ==========================================================================================
"""
#%% Linear (Multivariate) Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True, fit_intercept=True) # fit_intercept=True -> Include offset in model
print("params: ", model.coef_)
print("constant: ", model.intercept_)

#%% Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)
# --> Now we can apply linear regression to the transformend features
x_poly = poly_features.fit_transform(x)

# %% Logistic Regression (Is actually a classifier)
from sklearn.linear_model import LogisticRegression
model = LogisticRegpenalty={'l1', 'l2'}, C=1.0 # C is regularization strength

# %% Lasso and Ridge Regression
from sklearn.linear_model import Lasso, Ridge
model = Lasso(alpha=1.0) # alpha is regularization strength
model = Ridge(alpha=1.0) # alpha is regularization strength

# %% Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion={'gini', 'entropy', 'log_loss'}, 
                               max_depth=3, splitter={'best', 'random'}, 
                               random_state=42)

# %% RandomForest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion={'gini', 'entropy', 'log_loss'},
                               n_estimators=100, max_depth=3, 
                               random_state=42)

# %% Support Vector Machine
from sklearn.svm import SVC
model = SVC(kernel={'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'})

# %% k-Nearest-Neighbors
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

# %% Naive-Bayes-Classifier
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# %% Gaussian Process Regressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
kernel = 1.0 * RBF(length_scale=1.0)  # Radial Basis Function (RBF) kernel
gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)

# %% Model Trainging and Prediction ###
model.fit(x_train, y_train)
y_pred = model.predict(x_test) # to get predictions on the whole dataset
y_proba = model.predict_proba(x_test) # to get the probability for each class

# %% Metrics ###
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             roc_auc_score,
                             classification_report)

# Classification
print("Classification Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

# Regresssion
from sklearn.metrics import mean_squared_error, r2_score
print("Regression Metrics:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Classification Report
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

# Confusion Matrix Plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

classes=['no cancer', 'cancer']

fig, ax = plt.subplots(2, figsize=(6, 12))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes)
disp.plot(ax=ax[0])
ax[0].set_title("Confusion Matrix")
plt.show()

### Receiver Operating Curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve

# Function for plotting the ROC curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label = 'random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plot_roc_curve(fpr, tpr)

### Precision Recall Curve
# Function for plotting the Precision-Recall curve
def plot_rpc(recall, precision):
    plt.plot(recall, precision, color='orange', label='RPC')
    plt.ylabel('Precision')
    plt.xlabel('Recall = True Positive Rate')
    plt.title('Recall-Precision Curve')
    plt.legend()
    plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_proba)
plot_rpc(recall, precision)

# %%
"""
VALIDATION          ==========================================================================================
"""
# %% ### Cross-Validation ###
from sklearn.model_selection import cross_val_score

# z.B. with KNeighborsClassifier 
knn_cv = KNeighborsClassifier(n_neighbors=k)
# 10-fold cross-validation
cv_scores = cross_val_score(knn_cv, x_train, y_train, cv=10) 

# Leave One Out - crossvalidation (LOO)
from sklearn.model_selection import cross_val_score, LeaveOneOut
loo = LeaveOneOut()
cv_scores = cross_val_score(knn_cv, x_train, y_train, cv=loo) 

# %% ### Grid Search ###
from sklearn.model_selection import GridSearchCV

# z.B RandomForestClssifier
rf=RandomForestClassifier()
# Parameters that should be selected
params= {'n_estimators':[10,50,100,200],
         'max_depth':list(range(1,7))}

# scoring parameters default=accuracy for f1_score -> scoring='f1_macro'
grid = GridSearchCV(estimator=rf, param_grid=params, scoring='f1_macro', cv=10)
grid.fit(x_train, y_train)

score = grid.score(x_test, y_test) # Depends on scoring -> accuracy or f1-score

print(f'Score: {score:.2f}')
print(f'Best Paramenters {grid.best_params_}')

# %%
"""
UNSUPERVISED LEARNING  ==========================================================================================
"""
# %% ### Clustering Algorythms


from scipy.cluster.hierarchy import linkage
## Hierarchical clustering

# df = pd.DataFrame
row_clusters = linkage(df.values, method='complete', metric='euclidean')

# Plot (dendrogram)
from scipy.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters, 
                       labels=labels,
                       )
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()


## K-Means
from sklearn.cluster import KMeans

km = KMeans(n_clusters=8, 
            init='k-means++',  # Possible init = 'random' 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(x)
print(km.score(x))
print(km.inertia_)

# Plot Clusters (k-means)
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
ColorNames=list(colors.keys())
HSV=colors.values()
def PlotClusters(X,y, km):
    print("%i clusters" % km.n_clusters)
    plt.figure()
    for ClusterNumber in range(km.n_clusters):
        plt.scatter(X[y == ClusterNumber, 0],
                X[y == ClusterNumber, 1],
                s=50, c=ColorNames[ClusterNumber+1],
                marker='s', edgecolor='black',
                label='cluster {0}'.format(ClusterNumber+1))
    plt.scatter(km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()

PlotClusters(x, y_km, km)

## DBSCAN
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')   # eps: Search radius
y_db = db.fit_predict(x)
print("Number of clusters: {}".format(len(np.unique(y_db))))
print("Cluster sizes: {}".format(np.bincount(y_db + 1)))
plt.scatter(x[y_db == 0, 0], x[y_db == 0, 1],
            c='lightblue', marker='o', s=40,
            edgecolor='black', 
            label='cluster 1')
plt.scatter(x[y_db == 1, 0], x[y_db == 1, 1],
            c='red', marker='s', s=40,
            edgecolor='black', 
            label='cluster 2')
plt.legend()
plt.tight_layout()
plt.show()


# %% Elbow Curve / Silhouette score, BIC
distortions = []
## Elbow Curve (Example with k-Means) / BIC
ScoreList   = []
InertiaList = []
maxNumberOfClusters=15
n_samples=250
n_features=8
for k in range(1, maxNumberOfClusters):
    km = KMeans(n_clusters=k, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(x)
    distortions.append(km.inertia_)
    BIC=-km.score(x)/np.sqrt(n_samples)+np.log(n_samples)*n_features*k
    ScoreList.append(BIC)
    InertiaList.append(-km.score(x)/np.sqrt(n_samples))
   
plt.plot(range(1, maxNumberOfClusters), ScoreList, marker='^',label='BIC')
plt.plot(range(1, maxNumberOfClusters), InertiaList, marker='o',label='inertia')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# Function to calculate BIC / AIC for probabilistic model (Gauss)
km.

## Slihouett score
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(x, y_km, metric='euclidean')  # Example with k-Means clustering

# Plot
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()
print(silhouette_avg)

# %% Dimensionality Reduction: PCA tSNE
from sklearn.decomposition import PCA
## PCA

pca = PCA(n_components=100, whiten=True, random_state=0)
x_pca = pca.fit_transform(x_people)
#x_pca = pca.transform(x_people)

explained_variance_ratio = pca.explained_variance_ratio_

image_shape = people.images[0].shape
NumberOfSamples=x_pca.shape[0] # Number of pictures

fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})

for ix, target, ax in zip(np.arange(NumberOfSamples), y_people, axes.ravel()):
    image=np.reshape(pca.inverse_transform(x_pca[ix,:]),image_shape)
    ax.imshow(image)
    ax.set_title(str(y_people[ix])+': '+people.target_names[target])

## tSNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
x_reduced = tsne.fit_transform(x)

# Plot example for reduced mnist DataSet 3D->2D

plt.figure(figsize=(13,10))
plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y.astype(int), cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()




