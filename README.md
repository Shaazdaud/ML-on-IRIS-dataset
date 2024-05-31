# Machine Learning on the Iris Dataset

## Introduction

This project demonstrates the application of various machine learning algorithms on the Iris dataset. The Iris dataset is a classic dataset used for pattern recognition and classification, containing measurements of iris flowers from three different species. The project includes steps for data exploration, visualization, model training, evaluation, and comparison of different algorithms.

## Requirements

This project requires the following Python libraries:

- Python 3.x
- SciPy
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

Ensure that you have these libraries installed. You can install them using pip if necessary:

```bash
pip install scipy numpy matplotlib pandas scikit-learn
```

## Code Overview

### 1. Check Versions of Libraries

```python
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))
```

### 2. Load Libraries

```python
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
```

### 3. Load Dataset

```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())
```

### 4. Data Visualization

- **Box and Whisker Plots**

```python
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
```

- **Histograms**

```python
dataset.hist()
plt.show()
```

- **Scatter Plot Matrix**

```python
scatter_matrix(dataset)
plt.show()
```

### 5. Split-out Validation Dataset

```python
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
```

### 6. Spot Check Algorithms

```python
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
```

### 7. Make Predictions on Validation Dataset

```python
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
```

## Running the Code

1. Ensure you have all the required libraries installed.
2. Copy the code into a Python script or Jupyter Notebook.
3. Run the script. The script will output:
   - Versions of the libraries.
   - Dataset shape, head, description, and class distribution.
   - Data visualizations (box plots, histograms, scatter plot matrix).
   - Algorithm performance comparison.
   - Accuracy, confusion matrix, and classification report for the final model.

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
