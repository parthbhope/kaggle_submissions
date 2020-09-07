# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
tdata = data.iloc[:,2:]
y = data.iloc[:,1]

df = pd.DataFrame(tdata)
df = df.drop(columns=['Name','Ticket','Cabin','Embarked'])
pid = tdata.iloc[:,0]
df2 = pd.DataFrame(pid[:])
df2 = df.drop(829)
lb = LabelEncoder()
dd = df.iloc[:,:].values
dd[:,1] = lb.fit_transform(dd[:,1])
#dd[dd.
df1 = pd.DataFrame(dd)
imp = SimpleImputer()
imp = imp.fit(dd[:,2:3])
dd[:,2:3] = imp.transform(dd[:,2:3])
df1 = pd.DataFrame(dd)

dtree = DecisionTreeClassifier(criterion="entropy")
dtree.fit(dd,y)
pid = test.iloc[:,0]
test = test.iloc[:,1:]
df = pd.DataFrame(test)
df = df.drop(columns=['Name','Ticket','Cabin','Embarked'])
lb = LabelEncoder()
dd2 = df.iloc[:,:].values
df = pd.DataFrame(dd2)
dd2[:,1]= lb.fit_transform(dd2[:,1])
imp = SimpleImputer()
imp.fit(dd2[:,2:3])
dd2[:,2:3] = imp.transform(dd2[:,2:3])
imp = SimpleImputer()
imp = imp.fit(dd2[:,5:6])
dd2[:,5:6] = imp.transform(dd2[:,5:6])

df2 =pd.DataFrame(dd2)
res =dtree.predict(dd2)
df2 = pd.DataFrame(res)

dff = pd.DataFrame({'PassengerId':pid,'Survived':res})
dff.to_csv('predict.csv',index=False)