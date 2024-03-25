import pandas as pd
df = pd.read_csv('iris.csv')

# Let us check first and last few records of the dataframe
df.head()
df.tail()

# Now, let us inspect the columns and its datatype
df.info()

# Checking if the columns have null values
df.isnull().sum()

# Since the target column (species) contains multiple categories, we are going to encode it as follows
# Iris-setosa - 0, Iris-versicolor - 1, Iris-virginica-2

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['species'] = label.fit_transform(df['species'])
print(df)

# Storing features in variable 'X' and target varible in 'y'
X = df.drop(['species'], axis = 1)
print(X)

y = df['species']
print(y)

# Splitting the dataset into training and testing sets using scikit-learn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 60)

# Using Maching Learning Models to make predictions

# XGBoost
import xgboost as xgb

clf_x = xgb.XGBClassifier()
clf_x.fit(X_train,y_train)

pred_xgb = clf_x.predict(X_test)

# Let us now see how the classification report looks for the built model
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,pred_xgb))

cfm = confusion_matrix(y_test,pred_xgb)
print(cfm)

# Visualizing the confusion matrix with the help of seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cfm, annot= True, fmt ='g', xticklabels=['Iris-setosa','Iris-versicolor','Iris-virginica'],yticklabels=['Iris-setosa','Iris-versicolor','Iris-virginica'])
plt.ylabel('Predictions', fontsize=15)
plt.xlabel('Actual', fontsize=15)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


#  Naive Bayes - MultinomialNB and GaussianNB

from sklearn.naive_bayes import GaussianNB

clf_G = GaussianNB()
clf_G.fit(X_train,y_train)

pred_g = clf_G.predict(X_test)

print(classification_report(y_test,pred_g))

cfm_g = confusion_matrix(y_test,pred_g)
print(cfm_g)

from sklearn.naive_bayes import MultinomialNB

clf_M = MultinomialNB()
clf_M.fit(X_train,y_train)

pred_m = clf_M.predict(X_test)

print(classification_report(y_test,pred_m))

cfm_m = confusion_matrix(y_test,pred_m)
print(cfm_m)


# Support Vector Machine (SVM)
from sklearn.svm import SVC

clf_s = SVC()
clf_s.fit(X_train,y_train)

pred_s = clf_s.predict(X_test)

print(classification_report(y_test,pred_s))

cfm_s = confusion_matrix(y_test,pred_s)
print(cfm_s)

# Lets us now try different kernels of SVM using k-fold cross validation while keeping cross fold value as 10

from sklearn.model_selection import cross_val_score

L = ['linear','poly', 'sigmoid']

for i in L:
    classifier = SVC(kernel = i)
    classifier.fit(X,y)
    prediction = classifier.predict(X)
    score = cross_val_score(classifier, X,y,cv=10)
    print(score.mean())

# As we can see from above, Linear Kernel performs the best amongst the three kernels
    
from sklearn.neighbors import KNeighborsClassifier
clf_k = KNeighborsClassifier(n_neighbors= 8)
clf_k.fit(X_train,y_train)

pred_k = clf_k.predict(X_test)

print(classification_report(y_test,pred_k))

# Trying with different values of k ranging from 1 to 50
from sklearn.metrics import accuracy_score
for i in range(1,51,1):
    clfk = KNeighborsClassifier(n_neighbors=i)
    clfk.fit(X_train,y_train)
    predk = clfk.predict(X_test)
    scorek = accuracy_score(y_test,predk)
    print(scorek)

# Decision Tree

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

clf_d = DecisionTreeClassifier(random_state=125)
clf_d.fit(X_train,y_train)

pred_d = clf_d.predict(X_test)

print(classification_report(y_test,pred_d))

# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf_r = RandomForestClassifier(n_estimators=10, random_state=125)
clf_r.fit(X_train,y_train)

pred_r = clf_r.predict(X_test)

print(classification_report(y_test,pred_r))

