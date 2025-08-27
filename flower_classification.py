from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_decision_regions


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names= ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset= read_csv(url , names= names)

# print(dataset.head(20))
# print(dataset.describe())
# print(dataset.groupby('class').size())

# dataset.plot(kind='area' , subplots=True , layout=(2,2) , sharex=False , sharey=False)
# plt.show()

# scatter_matrix(dataset)
# plt.show()

array= dataset.values
# print(array)
X= array[: , 0:4]
# print(array)
y= array[: , 4]
X_train , X_validation  ,Y_train  , Y_validation = train_test_split( X , y , test_size=0.2 , random_state=1)

models=[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# print(models)

results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# plt.boxplot(results, labels=names)
# plt.title('Algorithm Comparison')
# plt.show()
# print(len(results[0]))

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_train)

print(accuracy_score(Y_train, predictions))
print(confusion_matrix(Y_train, predictions))
print(classification_report(Y_train, predictions))

le = LabelEncoder()
y_encoded = le.fit_transform(Y_train)

# Create a figure with three subplots
plt.figure(figsize=(20, 15))

# Plot 1: Sepal length vs Sepal width
plt.subplot(131)
X_train_2d = X_train[:, [0, 1]].astype(float)
model_2d = SVC(gamma='auto')
model_2d.fit(X_train_2d, y_encoded)
plot_decision_regions(X_train_2d, y_encoded, clf=model_2d, legend=2)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Decision Boundary: Sepal Features')

# Plot 2: Petal length vs Petal width
plt.subplot(132)
X_train_2d = X_train[:, [2, 3]].astype(float)
model_2d = SVC(gamma='auto')
model_2d.fit(X_train_2d, y_encoded)
plot_decision_regions(X_train_2d, y_encoded, clf=model_2d, legend=2)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Decision Boundary: Petal Features')

# Plot 3: Sepal length vs Petal length
plt.subplot(133)
X_train_2d = X_train[:, [0, 2]].astype(float)
model_2d = SVC(gamma='auto')
model_2d.fit(X_train_2d, y_encoded)
plot_decision_regions(X_train_2d, y_encoded, clf=model_2d, legend=2)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Decision Boundary: Sepal Length vs Petal Length')

plt.tight_layout()
plt.show()

# print(Y_train)