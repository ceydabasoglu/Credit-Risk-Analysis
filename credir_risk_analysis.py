import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("bankloan.csv")

data= data.dropna()

print(data.isnull().sum)

print(data.shape)

print(data.head())

print(data.describe())

print(data.info())

print(data['default'].value_counts())

sns.pairplot(data, hue='default', diag_kind='kde')
plt.show()

#korelasyon matrisini görüntüle
plt.figure(figsize=(12,12))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

for i in data.columns:
    plt.figure(figsize=(5,5))
    plt.hist(data=data, x=i)
    plt.xlabel(i)
    
    

def histplot(feature):
    fig,axs = plt.subplots(1,2, figsize=(15,5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=data, x=feature, bins = 30, kde=True, color='green')
    plt.subplot(1,2,2)
    sns.histplot(data=data, x= feature, bins = 30, kde = True, hue= 'default')
    plt.show()
for feature in data.columns:
    histplot(feature)


def kdeplot(feature):
    plt.figure(figsize=(10, 3))
    plt.title("Distribution for {}".format(feature))
    plot1 = sns.kdeplot(data[data['default'] == 0][feature].dropna(), color= 'grey')
    plot2 = sns.kdeplot(data[data['default'] == 1][feature].dropna(), color= 'Red')
    plt.legend(["NON-DEFAULTER","DEFAULTER"],loc='upper right')
    
for feature in data.columns:
    kdeplot(feature)
    

def distplot(feature):
    sns.distplot(data[feature].dropna())  
    plt.title("Distribution plot for {}".format(feature))
    plt.xlabel(feature)  
    plt.ylabel('Density') 
    
for feature in data.columns:
    distplot(feature)
    plt.show()  

sns.regplot(x='age', y='income', data=data)
plt.title('Regression Plot for Age vs Income')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()



sns.regplot(x='income', y='creddebt', data=data)
plt.title('Regression Plot for Income vs Creddebt')
plt.xlabel('Income')
plt.ylabel('Creddebt')
plt.show()


plt.scatter(x=data['age'], y=data['creddebt'])
plt.title('Scatter Plot for Age vs Creddebt')
plt.xlabel('Age')
plt.ylabel('Creddebt')
plt.show()


x = data.iloc[:, :6]
y = data['default']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size =0.2,random_state = 7)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#LOGİSTİC REGRESSION
log_reg_model = LogisticRegression()

log_reg_model.fit(x_train, y_train)

y_pred = log_reg_model.predict(x_test)

accuracy_log_reg = accuracy_score(y_test, y_pred)

print ("Logistic regression accuracy : ", accuracy_log_reg)


print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

weights = pd.Series(log_reg_model.coef_[0],
index=x.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))

#KNeibhoursClassifier
from sklearn.neighbors import KNeighborsClassifier
# K değerlerini ve bu K değerlerine karşılık gelen eğitim ve test doğruluklarını saklamak için boş listeler oluşturma
k_values = list(range(3, 50, 2))
train_accuracy = []
test_accuracy = []

# Farklı K değerleri için KNeighborsClassifier modelini oluşturma ve doğruluk değerlerini hesaplama
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    
    train_acc = accuracy_score(model.predict(x_train), y_train)
    test_acc = accuracy_score(model.predict(x_test), y_test)
    
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracy, 'ro-', label='Training Accuracy')
plt.plot(k_values, test_accuracy, 'bo-', label='Test Accuracy')
plt.title('Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# En iyi K değerini seçme
best_k = k_values[np.argmax(test_accuracy)]
print('Best K Value:', best_k)

# En iyi K değeri ile modeli tekrar eğitme
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(x_train, y_train)

y_pred_knn = best_model.predict(x_test)
test_accuracy_best = accuracy_score(y_test, y_pred_knn)
print('Accuracy with best K:', test_accuracy_best)
print(classification_report(y_test, y_pred_knn))

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier, plot_tree

model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)


model_gini.fit(x_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(model_gini, feature_names=x.columns.tolist(), class_names=['0', '1'], filled=True)
plt.title("Decision Tree")
plt.show()

y_pred_train_gini = model_gini.predict(x_train)
y_pred_test_gini = model_gini.predict(x_test)

train_accuracy_gini = accuracy_score(y_train, y_pred_train_gini)
test_accuracy_gini = accuracy_score(y_test, y_pred_test_gini)

print('Training-set accuracy score with gini criterion: {0:0.4f}'.format(train_accuracy_gini))
print('Test-set accuracy score with gini criterion: {0:0.4f}'.format(test_accuracy_gini))

print('Classification Report:')
print(classification_report(y_test, y_pred_test_gini))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_test_gini))


print('Accuracy :', accuracy_score(y_test, y_pred_test_gini))



#scaling
from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
x_scaled = scale.fit_transform(x)

#XG Boost

import time
from sklearn.model_selection import RepeatedKFold, cross_validate,GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0),
    'roc_auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovr')
}

params = {
    "XGB Classifier": {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'booster': ['gbtree', 'gblinear'],
        'gamma': [0, 0.5, 1],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [0.5, 1, 5],
        'base_score': [0.2, 0.5, 1]
    }
}

estimators = {'XGB Classifier': XGBClassifier(random_state=0)}

def test_algorithms(X, y):
    result = pd.DataFrame()
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
    for key, algorithm in estimators.items():
        start_time = time.time()
        print(f'{key}...', end='')
        model = algorithm
        cv_results = cross_validate(model, x_train, y, cv=cv, scoring=scoring, return_estimator=True)
        new_row = {
            'Algorithm': key,
            'accuracy': round(np.mean(cv_results['test_accuracy']), 5),
            'precision': round(np.mean(cv_results['test_precision']), 5),
            'recall': round(np.mean(cv_results['test_recall']), 5),
            'f1': round(np.mean(cv_results['test_f1']), 5),
            'roc_auc': round(np.mean(cv_results['test_roc_auc']), 5),
            'run_time': round((time.time() - start_time) / 60, 2),
            'model': cv_results['estimator'][0]
        }
        result = pd.concat([result, pd.Series(new_row)], axis=1)
        print(f'finished!!! {round((time.time() - start_time) / 60, 2)} min(s).')
    return result.transpose().sort_values(by='f1', ascending=False)

predicted_result = test_algorithms(x_train, y_train)
print("predicted result : ",predicted_result)
result = pd.concat([predicted_result], axis=0)
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
result['mean_metrics'] = result[metrics].mean(axis=1)
print(result[['Algorithm', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'run_time', 'mean_metrics']])
result.set_index(['Algorithm']).sort_values(by='precision', ascending=False).drop('model', axis=1)

#Bagging Classifier

from sklearn.ensemble import BaggingClassifier
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0),
    'roc_auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovr')
}


params = {
    "BaggingClassifier": {
        'base_estimator': [DecisionTreeClassifier()],
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0],
        'bootstrap': [True, False],
        'n_jobs': [-1],
        'random_state': [7]
    }
}


def test_algorithms(X, y):
    result = pd.DataFrame()
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
    model = BaggingClassifier()
    for key, param_grid in params.items():  
        start_time = time.time()
        print(f'{key}...', end='')
        model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit=False)
        model.fit(X, y)
        cv_results = model.cv_results_
        new_row = {
            'Algorithm': key,
            'accuracy': round(np.mean(cv_results['mean_test_accuracy']), 5),
            'precision': round(np.mean(cv_results['mean_test_precision']), 5),
            'recall': round(np.mean(cv_results['mean_test_recall']), 5),
            'f1': round(np.mean(cv_results['mean_test_f1']), 5),
            'roc_auc': round(np.mean(cv_results['mean_test_roc_auc']), 5),
            'run_time': round((time.time() - start_time) / 60, 2)
        }
        result = pd.concat([result, pd.Series(new_row)], axis=1)
        print(f'finished!!! {round((time.time() - start_time) / 60, 2)} min(s).')
    return result.transpose().sort_values(by='f1', ascending=False)

predicted_result = test_algorithms(x_train, y_train)
print(predicted_result)



