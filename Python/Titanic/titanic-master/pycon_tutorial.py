import pandas as pd
import numpy as np
df  = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/train.csv')
df.head(10)
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df.head(10)
df.info()
df = df.dropna()
df.info()
df['Sex'].unique()
df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)
df.info()
df['Embarked'].unique()
df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
df.info()
df = df.drop(['Sex', 'Embarked'], axis=1)
df.info()
df.head(10)
cols = df.columns.tolist()
print cols
cols = [cols[1]] + cols[0:1] + cols[2:]
print cols
df  = df[cols]
df.head(10)
df.info()
train_data = df.values
train_data.view()
train_data[1]
train_data[1][1]
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100)
RandomForestClassifier??
RandomForestClassifier??
model = model.fit(train_data[0:,2:], train_data[0:,0])
df_test = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/test.csv')
df_test.head(10)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test = df_test.dropna()
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
test_data = df_test.values
output = model.predict(test_data[:,1:])
train_data[1]
result = np.c_[test_data[:,0].astype(int), output.astype(int)]
result[1]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.head(10)
df_result.to_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/titanic_1-0.csv', index=False)
df_result.shape

"""
Didnt put the values in the NA fileds just removed the rows so the test data
has less rows to submit.


Now inserting the values in NA fieds
"""

dfa  = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/train.csv')
dfa.info()
df = dfa.drop(['Name', 'Ticket', 'Cabin'], axis=1)
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)
from scipy.stats import mode
mode_embarked = mode(df['Embarked'])[0][0]
mode_embarked
mode(df['Embarked'])[0]
mode(df['Embarked'])
mode(df['Embarked'])[0]
mode(df['Embarked'])[0][0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
df.info()
train_data = df.values
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_data[0:,2:],train_data[0:,0])
df_test = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/test.csv')
df_test.info()
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test.info()
age_mean = df_test['Age'].median()
age_mean = df['Age'].median()
df_test['Age'] = df_test['Age'].fillna(age_mean)
df_test.head(10)
fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
fare_means
fare_means[0]
fare_means[1]
df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)
df_test.info()
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
test_data = df_test.values
output = model.predict(test_data[:,1:])
result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/titanic_1-1.csv', index=False)
df_result.shape()

"""


For the column Embarked, however, replacing {C, S, Q} by {1, 2, 3} would seem to imply 
the ordering C < S < Q when in fact they are simply arranged alphabetically.

To avoid this problem, we create dummy variables. Essentially this involves creating new 
columns to represent whether the passenger embarked at C with the value 1 if true, 0 
otherwise. Pandas has a built-in function to create these columns automatically.

"""

dfa  = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/train.csv')
dfa.info()
df = dfa.drop(['Name', 'Ticket', 'Cabin'], axis=1)
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)
from scipy.stats import mode
mode_embarked = mode(df['Embarked'])[0][0]
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
pd.get_dummies(df['Embarked'], prefix='Embarked').head(10)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
df.info()
train_data = df.values
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_data[0:,2:],train_data[0:,0])
df_test = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/test.csv')
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test['Age'] = df_test['Age'].fillna(age_mean)
fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:fare_means[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],axis=1)
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
test_data = df_test.values
output = model.predict(test_data[:,1:])
result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/titanic_1-2.csv', index=False)


"""
Still only .75598


Now Parameter Tuning
"""

dfa  = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/train.csv')
df = dfa.drop(['Name', 'Ticket', 'Cabin'], axis=1)
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)
from scipy.stats import mode
mode_embarked = mode(df['Embarked'])[0][0]
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
pd.get_dummies(df['Embarked'], prefix='Embarked').head(10)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
df.info()

train_data = df.values

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
parameter_grid = {'max_features': [0.5, 1.],'max_depth': [5., None]}
grid_search = GridSearchCV(RandomForestClassifier(n_estimators = 100), parameter_grid,cv=5, verbose=3)
grid_search.fit(train_data[0:,2:], train_data[0:,0])
grid_search.grid_scores_

model = RandomForestClassifier(n_estimators = 100, max_features=0.5, max_depth=5.0)
model = model.fit(train_data[0:,2:],train_data[0:,0])
df_test = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/test.csv')
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test['Age'] = df_test['Age'].fillna(age_mean)
fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')

df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:fare_means[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],axis=1)
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
test_data = df_test.values
output = model.predict(test_data[:,1:])
result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/titanic_1-3.csv', index=False)



"""
Still Only .77990
"""

"""CROSS VALIDATION SET"""

dfa  = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/train.csv')
df = dfa.drop(['Name', 'Ticket', 'Cabin'], axis=1)
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)
from scipy.stats import mode
mode_embarked = mode(df['Embarked'])[0][0]
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
pd.get_dummies(df['Embarked'], prefix='Embarked').head(10)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
df.info()

train_data = df.values[:891]
X = train_data[:, 2:]
X.shape
y = train_data[:, 0]
y.shape
n = len(df)/2
X_train = X[:n, :]
y_train = y[:n]
X_test = X[n:, :]
y_test = y[n:]

"Prediction"
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model = model.fit(X_train, y_train)
y_prediction = model.predict(X_test)
print "prediction accuracy:", np.sum(y_test == y_prediction)*1./len(y_test)

"Swapping"
X_train, X_test = X_test, X_train
y_train, y_test = y_test, y_train
model = RandomForestClassifier(n_estimators=100)
model = model.fit(X_train, y_train)
y_prediction = model.predict(X_test)
print "prediction accuracy:", np.sum(y_test == y_prediction)*1./len(y_test)


"Built-in cross_validation method"
from sklearn.cross_validation import KFold
cv = KFold(n=len(train_data), n_folds=2)
for training_set, test_set in cv:
        X_train = X[training_set]
        y_train = y[training_set]
        X_test = X[test_set]
        y_test = y[test_set]
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        y_prediction = model.predict(X_test)
        print "prediction accuracy:", np.sum(y_test == y_prediction)*1./len(y_test)



"""See Plotting.py"""


"""Building Pipelines"""

import pandas as pd
import numpy as np
df  = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/train.csv')
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
age_mean = df['Age'].mean()

from scipy.stats import mode
mode_embarked = mode(df['Embarked'])[0][0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
pd.get_dummies(df['Embarked'], prefix='Embarked').head(10)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
df = df.fillna(-1)
train_data = df.values


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
imputer = Imputer(missing_values=-1)
classifier = RandomForestClassifier(n_estimators=100)
pipeline = Pipeline([
    ('imp', imputer),
    ('clf', classifier),
])
parameter_grid = {
	'imp__strategy': ['mean', 'median'],
    'clf__max_features': [0.5, 1],
    'clf__max_depth': [5, None],
}
grid_search = GridSearchCV(pipeline, parameter_grid, cv=5, verbose=3)
grid_search.fit(train_data[0::,1::], train_data[0::,0])
sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)
grid_search.best_score_
grid_search.best_params_

df['Age'].describe()
df['Age'] = df['Age'].map(lambda x: age_mean if x == -1 else x)
df['Age'].describe()
train_data = df.values

model = RandomForestClassifier(n_estimators = 100, max_features=0.5, max_depth=5)
model = model.fit(train_data[0:,2:],train_data[0:,0])
df_test = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/test.csv')
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test['Age'] = df_test['Age'].fillna(age_mean)
fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x: fare_means[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')], axis=1)
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
test_data = df_test.values
output = model.predict(test_data[:,1:])
result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/titanic_1-4.csv', index=False)



"""
Using Support Vector Machines
SVM
"""

import pandas as pd
import numpy as np
df  = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/train.csv')
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)

from scipy.stats import mode

mode_embarked = mode(df['Embarked'])[0][0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)

df = df.drop(['Sex', 'Embarked'], axis=1)

cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]

train_data = df.values

from sklearn.svm import SVC

model = SVC(kernel='linear')
model = model.fit(train_data[0:,2:], train_data[0:,0])

df_test = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/test.csv')

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Age'] = df_test['Age'].fillna(age_mean)

fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],
                axis=1)

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values

output = model.predict(test_data[:,1:])


result = np.c_[test_data[:,0].astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/titanic_2-1.csv', index=False)

"""
Only .760 

"""
"""
Now
SVM with Parameter Tuning
"""

import pandas as pd
import numpy as np
df  = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/train.csv')
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)
from scipy.stats import mode
mode_embarked = mode(df['Embarked'])[0][0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
train_data = df.values


from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
parameter_grid = {
    'C': [1., 10.],
    'gamma': [0.1, 1.]
}
grid_search = GridSearchCV(SVC(kernel='linear'), parameter_grid, cv=5, verbose=3)
grid_search.fit(train_data[0:,2:], train_data[0:,0])
sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)
grid_search.best_score_
grid_search.best_params_
model = SVC(kernel='linear', C=1.0, gamma=0.1)
model = model.fit(train_data[0:,2:], train_data[0:,0])


df_test = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/test.csv')
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test['Age'] = df_test['Age'].fillna(age_mean)
fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],
                axis=1)
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
test_data = df_test.values
output = model.predict(test_data[:,1:])
result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/titanic_2-2.csv', index=False)


"""Again Only .760"""

"""DEEP LEARNING TENSORFLOW AND SKFLOW"""

for k, N in enumerate(n_trials):
    sx = plt.subplot(len(n_trials) / 2, 2, k + 1)
    plt.xlabel("$p$, probability of heads") \
        if k in [0, len(n_trials) - 1] else None
    plt.setp(sx.get_yticklabels(), visible=False)
    heads = data[:N].sum()
    y = dist.pdf(x, 1 + heads, 1 + N - heads)
    plt.plot(x, y, label="observe %d tosses,\n %d heads" % (N, heads))
    plt.fill_between(x, 0, y, color="#348ABD", alpha=0.4)
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)