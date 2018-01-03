#/Titanic analysis Main file
#Gegaktakan
#7/9/2017

#https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    from collections import Counter
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    #print(outlier_indices)
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

import numpy as np   #Import matrix lib
import pandas as pd
import os            #import file naming tool
import csv           #CSV lib
import sys
import matplotlib.pyplot as plt
import sklearn

#open the train data fille and load into data
with open('train.csv') as csvfile_Part1:
   data1 = pd.read_csv(csvfile_Part1)
#    print data
#    print data.shape

### detect outliers from Age, SibSp , Parch and Fare
##Outliers_to_drop = detect_outliers(data1,2,["Age","SibSp","Parch","Fare"])
### Drop outliers
##data1 = data1.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


with open('test.csv') as csvfile_Part2:
   data2 = pd.read_csv(csvfile_Part2)

data = data1.append(data2,ignore_index=True)
#print(pd.isnull(data).sum())

#loops through each passenger and prints name
for row in range(0,data.PassengerId.count()):
   #print(row)
   #print(data.loc[row, 'Name'])
   nameSplit = data.loc[row, 'Name'].split(",")
   firstName = nameSplit[0]
   #print(firstName)
   secondNames = nameSplit[1]
   #print(secondNames)
   nameSplit2 = secondNames.split(". ")
   title = nameSplit2[0]
   # print(title)
   if(title == ' Mr'):
      data.loc[row, 'titles'] = 0
      #print(0)
   elif(title == ' Mrs'):
      data.loc[row, 'titles'] = 1
      #print(1)
   elif(title == ' Miss'):
      data.loc[row, 'titles'] = 2
      #print(2)
   elif(title == ' Rev'):
      data.loc[row, 'titles'] = 3
      #print(3)
   elif(title == ' Master'):
      data.loc[row, 'titles'] = 3
      #print(4)
   elif(title == ' Dr'):
      data.loc[row, 'titles'] = 3
      #print(5)   
   elif(title == ' the Countess'):
      data.loc[row, 'titles'] = 3
      #print(6)
   elif(title == ' Col'):
      data.loc[row, 'titles'] = 3
      #print(7)
   elif(title == ' Mlle'):
      data.loc[row, 'titles'] = 2
      #print(2)
   elif(title == ' Sir'):
      data.loc[row, 'titles'] = 3
      #print(8)
   elif(title == ' Capt'):
      data.loc[row, 'titles'] = 3
      #print(9)
   elif(title == ' Don'):
      data.loc[row, 'titles'] = 3
      #print(10)
   elif(title == ' Dona'):
      data.loc[row, 'titles'] = 3
      #print(12)
   elif(title == ' Jonkheer'):
      data.loc[row, 'titles'] = 3
      #print(13)
   elif(title == ' Lady'):
      data.loc[row, 'titles'] = 3
      #print(14)
   elif(title == ' Major'):
      data.loc[row, 'titles'] = 3
      #print(15)
   elif(title == ' Ms'):
      data.loc[row, 'titles'] = 2
      #print(2)
   elif(title == ' Mme'):
      data.loc[row, 'titles'] = 1
      #print(1)
   #print (data.loc[row,:])
   #break
##^^could need to collect rare titles
#print data
      
data.loc[0:data.PassengerId.count(),'level'] = 0      
#loops through each passenger and prints cabin info
for row in range(0,data.PassengerId.count()):
   deck = str(data.loc[row, 'Cabin'])
   #print(deck)
   if('A' == deck[0]):
      data.loc[row, 'level'] = 1
      #print(1)
   elif('B' == deck[0]):
      data.loc[row, 'level'] = 2
      #print(2)
   elif('C' == deck[0]):
      data.loc[row, 'level'] = 3
      #print(3)
   elif('D' == deck[0]):
      data.loc[row, 'level'] = 4
      #print(4)
   elif('E' == deck[0]):
      data.loc[row, 'level'] = 5
      #print(5)
   elif('F' == deck[0]):
      data.loc[row, 'level'] = 6
      #print(6)
   else:
      data.loc[row, 'level'] = 0
#      print(0)
#print data
      
data.loc[0:data.PassengerId.count(),'gender'] = 0 
#loops through each passenger and prints sex
for row in range(0,data.PassengerId.count()):
   sex = str(data.loc[row, 'Sex'])
   #print(sex)
   if('male' == sex):
      data.loc[row, 'gender'] = 0
      #print(0)
   elif('female' == sex):
      data.loc[row, 'gender'] = 1
      #print(1)

##
###ticket split seems to make it worse
##TicketPREF = []
##data.loc[0:data.PassengerId.count(),'TicketPref'] = 0
##for i in range(0,data.PassengerId.count()):
##   ticket = data.loc[i,'Ticket']
##   if not ticket.isdigit() :
##      TicketPREF.append(ticket.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
##      A = ticket.replace(".","").replace("/","").strip().split(' ')[0]
##   else:
##      TicketPREF.append("X")        
##      A = "X"
###  print(A)
##   if('A5' == A):
##      data.loc[i, 'TicketPref'] = 0
###      print(0)
##   elif('PC' == A):
##      data.loc[i, 'TicketPref'] = 1
## #     print(1)
##   elif('STONO2' == A):
##      data.loc[i, 'TicketPref'] = 2
##  #    print(2)
##   elif('PP' == A):
##      data.loc[i, 'TicketPref'] = 3
##   #   print(3)
##   elif('CA' == A):
##      data.loc[i, 'TicketPref'] = 4
##    #  print(4)
##   elif('SOC' == A):
##      data.loc[i, 'TicketPref'] = 5
##     # print(5)
##   elif('SOTONOQ' == A):
##      data.loc[i, 'TicketPref'] = 6
##      #print(6)
##   elif('FC' == A):
##      data.loc[i, 'TicketPref'] = 7
###      print(7)
##   else:
##      data.loc[i, 'TicketPref'] = 8
## #     print(8)
##print(data.loc[0:data.PassengerId.count(),'TicketPref'])
##sys.exit
###print(data.loc[0:data.PassengerId.count(),'Ticket'])
##
##
      
TicketNum = []
data.loc[0:data.PassengerId.count(),'TicketNum'] = 0
for row in range(0,data.PassengerId.count()):
   ticket = data.loc[row,'Ticket']
   if ticket.isdigit() :
      data.loc[0:data.PassengerId.count(),'TicketNum'] = ticket.split(' ')[len(ticket.split(' '))-1]
   else:
      data.loc[0:data.PassengerId.count(),'TicketNum'] = 0

      
      
data.loc[0:data.PassengerId.count(),'emabarkation'] = 0
#embarked based on class, title and cost
#assumed C for blanks based on https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic analysis
for row in range(0,data.PassengerId.count()):
   location = str(data.loc[row, 'Embarked'])
   #print(location)
   if('S' == location):
      data.loc[row, 'emabarkation'] = 0
      #print(0)
   elif('Q' == location):
      data.loc[row, 'emabarkation'] = 1
      #print(1)
   else:
      data.loc[row, 'emabarkation'] = 2
      #print(2)


#use average fare by default (could be improved by rpart, but not sure where that is/ how to use in python
data['Fare'].fillna(value=(data['Fare'].median()), inplace=True)
data["Fare"] = data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

data.loc[0:data.PassengerId.count(),'Family'] = 0
#use Family size general sizes to eliminate overfitting training data
for row in range(0,data.PassengerId.count()):
   par_ch = data.loc[row, 'SibSp']
   Sib_Sp = data.loc[row, 'Parch']
   #print(location)
   if((par_ch+Sib_Sp) == 0):
      data.loc[row, 'Family'] = 0
      #print(0)
   elif((par_ch+Sib_Sp) <= 3):
      data.loc[row, 'Family'] = 1
      #print(1)
   elif((par_ch+Sib_Sp) <= 4):
      data.loc[row, 'Family'] = 1
      #print(1)
   else:
      data.loc[row, 'Family'] = 2
      #print(2)

#age based on sex, title, class, cabin sib, and parch
from sklearn import tree
fulldata1 = data.loc[:,:].drop('Name', axis=1).drop('Sex', axis=1).drop('Ticket', axis=1).drop('Cabin', axis=1).drop('Embarked', axis=1).drop('Survived', axis=1).drop('Parch', axis=1).drop('SibSp', axis=1)
ageless = fulldata1[np.isnan(fulldata1.Age)]
ageMarked = fulldata1[1 != np.isnan(fulldata1.Age)]
x = ageMarked.drop('Age', axis=1)
y = ageMarked.loc[:,'Age'].astype(int)
testData = ageless.drop('Age', axis=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
AgePrediction = clf.predict(testData)
#print(AgePrediction)
agePredIndex = 0
for row in range(len(data)):
   #print(row)
   if(np.isnan(data.loc[row, 'Age'])):
      data.loc[row,'Age'] = (AgePrediction[agePredIndex])
      agePredIndex +=1



from sklearn.model_selection import train_test_split
fulldata = data.loc[:(len(data1)-1),:].drop('Name', axis=1).drop('Sex', axis=1).drop('Ticket', axis=1).drop('Cabin', axis=1).drop('Embarked', axis=1).drop('Parch', axis=1).drop('SibSp', axis=1)
#train on data to predict survival random forest
#print(len(data))
#print(pd.isnull(fulldata).sum())
x = fulldata.drop('Survived', axis=1)
y = fulldata.loc[:,'Survived']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25, random_state=12342)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)

if(len(x_test)>0):
   pred=rf.predict(x_test)
   sol = y_test.values
   count = 0
   for row in range(len(pred)):
      if pred[row]== sol[row]:
         count+=1

   print(count)
   print(len(pred))
   per = float(count)/len(pred)      
   print('percent correct randomForestClassifier: {0}'.format(per))

#_____________________________________________________________________
predData = data.loc[len(data1):len(data),:].drop('Name', axis=1).drop('Sex', axis=1).drop('Ticket', axis=1).drop('Cabin', axis=1).drop('Embarked', axis=1).drop('Parch', axis=1).drop('SibSp', axis=1)
predData = predData.drop('Survived', axis=1)
#print(pd.isnull(predData).sum())
prediction = rf.predict(predData).astype(int)
Ind = range(len(data1)+1,len(data)+1)
#print(len(data))
#print(prediction)
submittal = pd.DataFrame({'PassengerId' : pd.Series(Ind, Ind)})
submittal['Survived'] = prediction
#print (submittal)

submittal.to_csv('Results',index=False)









