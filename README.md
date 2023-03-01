# SPH6004 Assignment1
 Code for Assignment 1
 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

pd.set_option('display.max_rows', 500)

data = pd.read_csv('Assignment_1_data.csv', header = 0)
print(data.shape)
print(list(data.columns))

data_dummies = pd.get_dummies(data, columns=['gender'])
data_dummies = data_dummies.drop('gender_M',axis=1)
data_dummies.head(10)
data_dummies.dtypes

data_dummies['outcome'].value_counts()

count_no_int = len(data_dummies[data_dummies['outcome']==0])
count_int = len(data_dummies[data_dummies['outcome']==1])
pct_of_int = count_int/(count_no_int+count_int)
print("percentage of intubation is", pct_of_int*100)

data_dummies.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')

data_dummies.groupby('gender_F').mean()

data_dummies['heart_rate_min'] = data_dummies['heart_rate_min'].fillna(data_dummies['heart_rate_min'].median())
data_dummies['heart_rate_max'] = data_dummies['heart_rate_max'].fillna(data_dummies['heart_rate_max'].median())
data_dummies['heart_rate_mean'] = data_dummies['heart_rate_mean'].fillna(data_dummies['heart_rate_mean'].median())
data_dummies['mbp_min'] = data_dummies['mbp_min'].fillna(data_dummies['mbp_min'].median())
data_dummies['mbp_max'] = data_dummies['mbp_max'].fillna(data_dummies['mbp_max'].median())
data_dummies['mbp_mean'] = data_dummies['mbp_mean'].fillna(data_dummies['mbp_mean'].median())
data_dummies['sbp_min'] = data_dummies['sbp_min'].fillna(data_dummies['sbp_min'].median())
data_dummies['sbp_max'] = data_dummies['sbp_max'].fillna(data_dummies['sbp_max'].median())
data_dummies['sbp_mean'] = data_dummies['sbp_mean'].fillna(data_dummies['sbp_mean'].median())
data_dummies['dbp_min'] = data_dummies['dbp_min'].fillna(data_dummies['dbp_min'].median())
data_dummies['dbp_max'] = data_dummies['dbp_max'].fillna(data_dummies['dbp_max'].median())
data_dummies['dbp_mean'] = data_dummies['dbp_mean'].fillna(data_dummies['dbp_mean'].median())
data_dummies['temperature_min'] = data_dummies['temperature_min'].fillna(data_dummies['temperature_min'].median())
data_dummies['temperature_max'] = data_dummies['temperature_max'].fillna(data_dummies['temperature_max'].median())
data_dummies['temperature_mean'] = data_dummies['temperature_mean'].fillna(data_dummies['temperature_mean'].median())
data_dummies['glucose_min'] = data_dummies['glucose_min'].fillna(data_dummies['glucose_min'].median())
data_dummies['glucose_max'] = data_dummies['glucose_max'].fillna(data_dummies['glucose_max'].median())
data_dummies['wbc_min'] = data_dummies['wbc_min'].fillna(data_dummies['wbc_min'].median())
data_dummies['wbc_max'] = data_dummies['wbc_max'].fillna(data_dummies['wbc_max'].median())
data_dummies['creatinine_min'] = data_dummies['creatinine_min'].fillna(data_dummies['creatinine_min'].median())
data_dummies['creatinine_max'] = data_dummies['creatinine_max'].fillna(data_dummies['creatinine_max'].median())
data_dummies['hemoglobin_min'] = data_dummies['hemoglobin_min'].fillna(data_dummies['hemoglobin_min'].median())
data_dummies['hemoglobin_max'] = data_dummies['hemoglobin_max'].fillna(data_dummies['hemoglobin_max'].median())
data_dummies['urineoutput'] = data_dummies['urineoutput'].fillna(data_dummies['urineoutput'].median())
data_dummies['sofa_cardiovascular'] = data_dummies['sofa_cardiovascular'].fillna(data_dummies['sofa_cardiovascular'].median())
data_dummies['sofa_cns'] = data_dummies['sofa_cns'].fillna(data_dummies['sofa_cns'].median())
data_dummies['sofa_renal'] = data_dummies['sofa_renal'].fillna(data_dummies['sofa_renal'].median())
data_dummies['sofa_coagulation'] = data_dummies['sofa_coagulation'].fillna(data_dummies['sofa_coagulation'].median())

data_dummies

data_dummies.isna()

data_dummies.isna().sum()

data_dummies = data_dummies.dropna(axis='columns')

X = data_dummies.loc[:, data_dummies.columns != 'outcome']
y = data_dummies.loc[:, data_dummies.columns == 'outcome']

data_dummies.outcome.astype(int)

y.describe()

 !pip install imbalanced-learn
 
 from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['outcome'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no intubation in oversampled data",len(os_data_y[os_data_y['outcome']==0]))
print("Number of intubation",len(os_data_y[os_data_y['outcome']==1]))
print("Proportion of no intubation data in oversampled data is ",len(os_data_y[os_data_y['outcome']==0])/len(os_data_X))
print("Proportion of intubation data in oversampled data is ",len(os_data_y[os_data_y['outcome']==1])/len(os_data_X))

X.head()

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# data = pd.read_csv("Assignment_1_data.csv")

from sklearn.model_selection import train_test_split

# X = data.drop("outcome", axis=1)
# y = data["outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

##Forward selection
def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.Logit(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features
    
   forward_selection(X,y)
   
   best_features = ['sofa_cns',
 'heart_rate_min',
 'urineoutput',
 'gender_F',
 'age',
 'dbp_min',
 'charlson_comorbidity_index',
 'temperature_max',
 'glucose_min',
 'mbp_max',
 'dbp_max']

#Logistic regression
X=os_data_X[best_features]
y=os_data_y['outcome']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# data = pd.read_csv("Assignment_1_data.csv")

from sklearn.model_selection import train_test_split

# X = data.drop("outcome", axis=1)
# y = data["outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrfrom sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

def backward_elimination(data, target,significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.Logit(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features
    
    backward_elimination(X,y)
    
    features = ['age',
 'heart_rate_min',
 'heart_rate_mean',
 'mbp_min',
 'mbp_mean',
 'sbp_max',
 'sbp_mean',
 'dbp_mean',
 'temperature_mean',
 'glucose_min',
 'glucose_max',
 'urineoutput',
 'sofa_cns',
 'charlson_comorbidity_index',
 'gender_F']

#Logistic regression
X=os_data_X[features]
y=os_data_y['outcome']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# data = pd.read_csv("Assignment_1_data.csv")

from sklearn.model_selection import train_test_split

# X = data.drop("outcome", axis=1)
# y = data["outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

