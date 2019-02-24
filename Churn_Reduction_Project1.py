
#Load libraries



import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import KNN
from scipy.stats import chi2_contingency
sns.set()


# Load data


train = pd.read_csv("Train_data.csv")

test = pd.read_csv("Test_data.csv")


# Combine Train and test data


train_test = pd.concat([train, test], axis=0)




########## Exploratory Data Anlysis  ##############
train_test.head()


#Check statistical description of the dataset
train_test.describe()

#Check datatypes and missing values
train_test.info()

#Check ratio of target variable
sns.countplot(train_test['Churn'])

#Check number of unique values for each column
for i in train_test.columns:
    print(str(i)+"["+str(train_test[i].dtypes)+"]"+" : "+str(len(train_test[i].unique())))


#Since phone number has all unique values, it won't add much value to predict dependent variable. So let's drop that column
train_test.drop('phone number', axis=1, inplace =True)



#Save categorical variables and continious variales in seperate list
catnames=[]
cnames=[]
for i in range(0, train_test.shape[1]):
    if(train_test.iloc[:,i].dtypes != 'object'):
        cnames.append(train_test.columns[i])
    else:
        catnames.append(train_test.columns[i])





#State Mapping and get them to similar range as other feature
c=0
M={}
for k in train_test['state'].unique():
    M[k]=c
    c+=0.1
    c=round(c,2)


train_test['state'] = train_test['state'].map(M)



#Objects to Numerical as State already mapped
features_cov = ['international plan', 'voice mail plan', 'Churn']


#Convert above mentioned catogorical variables to numeric

for i in features_cov:
    train_test.loc[:,i] = pd.Categorical(train_test.loc[:,i])
    train_test.loc[:,i] = train_test.loc[:,i].cat.codes



#Convert into proper datatypes
for i in catnames:
    train_test.loc[:,i] = train_test.loc[:,i].astype('object')


# In[12]:


#Since Area code has only 3 unique values, map them to respective normalized values
train_test.loc[:,'area code'] = pd.Categorical(train_test.loc[:,'area code'])
train_test.loc[:,'area code'] = train_test.loc[:,'area code'].cat.codes 




#Since number customer service calls could be categorised as it could be mapped  into small number of buckets
c=0
M={}
for k in train_test['number customer service calls'].unique():
    M[k]=c
    c+=0.5
    c=round(c,2)


train_test['number customer service calls'] = train_test['number customer service calls'].map(M)


train_test.loc[:,'number customer service calls'] = train_test.loc[:,'number customer service calls'].astype('object')




#Since number vmail messages could be categorised, it is mapped  into small number of buckets
c=0
M={}
for k in train_test['number vmail messages'].unique():
    M[k]=c
    c+=0.1
    c=round(c,2)


train_test['number vmail messages'] = train_test['number vmail messages'].map(M)



train_test.loc[:,'number vmail messages'] = train_test.loc[:,'number vmail messages'].astype('object')


#update Categorical and continious variable list


catnames=[]
cnames=[]
for i in range(0, train_test.shape[1]):
    if(train_test.iloc[:,i].dtypes != 'object'):
        cnames.append(train_test.columns[i])
    else:
        catnames.append(train_test.columns[i])



################## Outlier Analysis ####################


# Boxplot to visualise outliers for each predictot in comparion with target variable
for i in cnames:
    plt.figure()
    sns.boxplot(y=i, data = train_test, x = 'Churn')

    
#Detect Outliers for dataset and replace it with NA and impute
for i in cnames:
    #print(i)
    q75, q25 = np.percentile(train_test.loc[:,i], [75 ,25])
    iqr = q75 - q25

    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    
##    print(minimum)
##    print(maximum)
##    
##    print(len(train_test[train_test.loc[:,i] < minimum]))
##    print(len(train_test[train_test.loc[:,i] > maximum]))
    
    train_test[i][train_test.loc[:,i] < minimum] = np.nan
    train_test[i][train_test.loc[:,i] > maximum] = np.nan

#Impute NA with KNN_Imputation
train_test = pd.DataFrame(KNN(k = 3).complete(train_test), columns = train_test.columns)


################## Feature Selection #########################

#Detect correlation between continious variables
df_corr = train.loc[:,cnames]


#plot with a heat map

f,ax = plt.subplots(figsize = (7,5))

corr = df_corr.corr()

sns.heatmap(corr)


#remove 'Churn' from catnames list as it will be used for chi-square further
catnames.remove('Churn')
catnames


#Features/variables to be dropped which are selected from correlation matrix
to_drop = ['total day charge', 'total eve charge', 'total night charge', 'total intl charge']



#Chi-squared test of independence

for i in catnames:

    chi2, p, dof, ex = chi2_contingency(pd.crosstab(train_test['Churn'],train_test[i]))
    if(p > 0.05):
        print(i)
        print(p)
        to_drop.append(i)


#Dimensionality Reduction, drop selected above variables to be dropped
train_test = train_test.drop(to_drop, axis = 1)



#Remove dropped variables from the continious variables list
for i in to_drop:
    if i in cnames:
        cnames.remove(i)



################# Feature Scaling #######################

#normalization

for i in cnames:
    print(i)
    train_test[i] = (train_test[i] - np.min(train_test[i])) / (np.max(train_test[i]) - np.min(train_test[i]))



################ Model Development ###################

 
#import necessary libraries from sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


#split processed data into train and test
train = train_test.iloc[0:3333,:]

test = train_test.iloc[3333:,:].reset_index(drop=True)


#split into model feedable format

X_train = train.drop('Churn', axis = 1)
y_train = train['Churn']
X_test = test.drop('Churn', axis = 1)
y_test = test['Churn']


#### Decision Tree Classifier


clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X_train, y_train)

DT_predictions = clf.predict(X_test)


print("DT_accuracy: {}".format(accuracy_score(y_test, DT_predictions)))


#Confusion Matrix
conf_mat = pd.crosstab(y_test, DT_predictions)


TN = conf_mat.iloc[0,0] 
FP = conf_mat.iloc[0,1]
FN = conf_mat.iloc[1,0]
TP = conf_mat.iloc[1,1]


accuracy = ((TP+TN)/(TN+FP+FN+TP))*100

#FNR
FNR = FN*100/(FN+TP)
print("******* Decision Tree ********")
print(accuracy)
print(FNR)



########### Random Forest Classifier

RF_clf = RandomForestClassifier(n_estimators=500)
RF_clf.fit(X_train, y_train)
RF_predictions = RF_clf.predict(X_test)


conf_mat = pd.crosstab(y_test,RF_predictions)


TN = conf_mat.iloc[0,0] 
FP = conf_mat.iloc[0,1]
FN = conf_mat.iloc[1,0]
TP = conf_mat.iloc[1,1]


print("******* Random Forest *********")
accuracy = ((TP+TN)/(TN+FP+FN+TP))*100
print(accuracy)

FNR = FN*100/(FN+TP)
print(FNR)


########## Naive Bayes


NB_clf = GaussianNB()

NB_clf.fit(X_train, y_train)

NB_prediction = NB_clf.predict(X_test)

conf_mat = pd.crosstab(y_test, NB_prediction)

TN = conf_mat.iloc[0,0] 
FP = conf_mat.iloc[0,1]
FN = conf_mat.iloc[1,0]
TP = conf_mat.iloc[1,1]

print("****Naive Bayes*******")
accuracy = ((TP+TN)/(TN+FP+FN+TP))*100
print(accuracy)

FNR = FN*100/(FN+TP)
print(FNR)



########### KNN

#Check the right value of n-neighbours for KNN

acc = []


for i in range(1,40):
    #print("n_estimators : {}".format(i))
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_prediction = knn.predict(X_test)
    #print(accuracy_score(y_test, knn_prediction))
    acc.append(accuracy_score(y_test, knn_prediction))



plt.figure(figsize=(10,6))
plt.plot(acc, marker='o', markerfacecolor = 'red')
plt.savefig("knn.png")




knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)
#print(accuracy_score(y_test, knn_prediction))

conf_mat = pd.crosstab(y_test, knn_prediction)

TN = conf_mat.iloc[0,0] 
FP = conf_mat.iloc[0,1]
FN = conf_mat.iloc[1,0]
TP = conf_mat.iloc[1,1]


print("********* KNN ************")
accuracy = ((TP+TN)/(TN+FP+FN+TP))*100
print(accuracy)

FNR = FN*100/(FN+TP)
print(FNR)


################# Logistic Regression


import statsmodels.api as sm

logit = sm.Logit(y_train, X_train).fit()

logit.summary()


test['a'] = logit.predict(X_test)

test['act'] = 1

test.loc[test.a < 0.5, 'act'] = 0

CM = pd.crosstab(y_test, test['act'])

TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

print("******** Logistic Regression ***********")
accuracy = ((TP+TN)*100)/(TP+TN+FP+FN)
print(accuracy)

FNR = (FN*100)/(FN+TP)
print(FNR)

