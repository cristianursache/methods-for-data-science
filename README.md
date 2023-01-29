Implementations of various ML algorithms and methods (using numpy only) and their applications.

## 1. Supervised learning

### 1.1. Regression task: linear regression, ridge regression, relaxation of LASSO regression.

`Linear regression` - implemented using normal equations.

`Ridge regression` - implemented using normal equations; includes cross-validation for hyperparameter optimisation ( $\lambda$ ).

`Relaxation of LASSO regression` - implemented using gradient descent; includes cross-validation for hyperparameter optimisation ( $\lambda$ ).

**Use case**: predict toxicity level of chemical substances (continuous target).

### 1.2. Classification task: kNN, random forest, SVM.

`kNN` - implemented using Euclidean distance as distance function; includes cross-validation for hyperparameter optimisation ( $k$ ).

`Random forest` - implemented using cross-entropy as loss function; includes cross-validation for hyperparameter optimisation (number of trees, max depth).

`SVM` - implemented using SGD; includes cross-validation for hyperparameter optimisation ( $\lambda$ ) and ROC curve.

**Use case**: predict whether breast tumours are benign or malignant (binary target).

### 1.3 Multi-class classification task: MLP, CNN.

`MLP` - implemented using SGD for forward and backward propagation and cross-entropy as a loss function.

`CNN` - implemented using tensorflow; includes cross-validation for hyperparameter optimisation (dropout rate).

**Use case**: predict class of items sold by an online fashion store (Fashion-MNIST data set).
    
## 2. Unsupervised learning

### 2.1. Dimensionality reduction: PCA.

**Use case**: dimensionality reduction (to two dimensions) of Fashion-MNIST data set.

### 2.2. Clustering: k-means clustering, hierarchical clustering.

`k-means clustering` - used within-cluster distance to determine optimal number of clusters.

**Use case**: cluster the reduced space of Fashion-MNIST data set.

`Hierarchical clustering` - used Silhouette score to determine optimal number of clusters.

**Use case**: cluster the feature matrix of bottlenose dolphins data set.
