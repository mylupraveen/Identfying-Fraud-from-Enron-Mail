#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


from tester import dump_classifier_and_data,test_classifier
from feature_format import featureFormat, targetFeatureSplit

import operator
from sklearn.feature_selection import SelectKBest,f_classif

import matplotlib.pyplot
from sklearn import naive_bayes 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit






### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

POI_label = ['poi']

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                      'expenses','exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
                      'director_fees']          

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi']


features_list = POI_label + financial_features + email_features

print 'Total No. of features are', len(features_list)                      # number of features used

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
print 'Number of data points are ', len(data_dict)                         # total number of data points


poi = 0                                                                    # allocation across classes (POI/non-POI)
for person in data_dict:
    if data_dict[person]['poi'] == True:
       poi += 1
print("Total number of poi: %i" % poi)
print("Total number of non-poi: %i" % (len(data_dict) - poi))




### Task 2: Remove outliers

def Outliers(my_dataset, feature_x, feature_y):
    
    data = featureFormat(my_dataset, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter( x, y )
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()

print(Outliers(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))  # Identifying outliers
print(Outliers(data_dict, 'salary', 'bonus'))
print(Outliers(data_dict, 'total_payments', 'total_stock_value'))
identity = []

for person in data_dict:
    if data_dict[person]['total_payments'] != "NaN":
        identity.append((person, data_dict[person]['total_payments']))
print("Outlier:")
print(sorted(identity, key = lambda x: x[1], reverse=True)[0:4])

fi_nan_dict = {}
for person in data_dict:
    fi_nan_dict[person] = 0
    for feature in financial_features:                         # Persons with financial features as NaN
        if data_dict[person][feature] == "NaN":
            fi_nan_dict[person] += 1
sorted(fi_nan_dict.items(), key=lambda x: x[1])

email_nan_dict = {}
for person in data_dict:
    email_nan_dict[person] = 0
    for feature in email_features:
        if data_dict[person][feature] == "NaN":                # Persons with email features as NaN
            email_nan_dict[person] += 1
sorted(email_nan_dict.items(), key=lambda x: x[1])

data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)              # Removing Outliers
data_dict.pop("LOCKHART EUGENE E", 0)





### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


def computeFraction( poi_messages, all_messages ):
    if poi_messages=='NaN' or all_messages=='NaN':
        return 'NaN'
    else:
        return float(poi_messages)/all_messages


for keys,features in data_dict.items():
    a = features['from_poi_to_this_person']
    b = features['from_messages']
    c = features['from_this_person_to_poi']
    d = features['to_messages']
    e = features['total_payments']
    f = features['total_stock_value']
    g = features['bonus']
    h = features['salary']

    features['fraction_from_poi'] = computeFraction(a,b)
    features['fraction_to_poi'] = computeFraction(c,d) 
    
    if a == 'NaN' or b == 'NaN':
        features['total_net_worth'] = 'NaN'
    else:
        features['total_net_worth'] = a + b  
    
    if e == 'NaN' or f == 'NaN':
        features['bonus_salary_ratio'] = 'NaN'
    else :
        features['bonus_salary_ratio'] = float(e) /float(f)   

    
features_list += ['total_net_worth'] + ['bonus_salary_ratio']+\
['fraction_from_poi'] + ['fraction_to_poi']

features_list.remove('email_address')                                     # eliminate feature email address
print "total number of features:", len(features_list)," "

my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# using SelectKBest
features_selected=[]
clf = SelectKBest(f_classif,k=5)
selected_features = clf.fit_transform(features,labels)
for i in clf.get_support(indices=True):
    features_selected.append(features_list[i+1])
features_score = zip(features_list[1:25],clf.scores_[:24])
features_score = sorted(features_score,key=operator.itemgetter(1),reverse=True)

features_list = ['poi']+features_selected
print "Scores of the features :\n"
for i in features_score:
    print i
print features_list



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

print "GaussianNB classifier(Default) :"
clf1 = GaussianNB()
test_classifier(clf1, my_dataset, features_list, folds = 1000)


print "LinearSVC classifier(default)"
clf2 = LinearSVC()
test_classifier(clf2, my_dataset, features_list, folds = 1000)


print "Decission Tree classifier(default) :"
clf3 = DecisionTreeClassifier()
test_classifier(clf3, my_dataset, features_list, folds = 1000)


print "Kneighbour classifier(default)"
clf4 = KNeighborsClassifier()
test_classifier(clf4, my_dataset, features_list, folds = 1000)




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#%%
# Example starting point. Try investigating other evaluation techniques!


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
shuffle= StratifiedShuffleSplit(labels_train,n_iter = 25,test_size = 0.5,
                                random_state = 0)
param_grid = {
         'pca__n_components':[1, 2, 3, 4, 5, 6]
          }
estimators = [('pca',PCA()),('gaussian',GaussianNB())]
pipe = Pipeline(estimators)
gs = GridSearchCV(pipe, param_grid,n_jobs = 1,scoring = 'f1',cv = shuffle)
gs.fit(features_train,labels_train)
pred = gs.predict(features_test)
clf = gs.best_estimator_
test_classifier(clf, my_dataset, features_list, folds = 1000)
print "best parameters ",gs.best_params_
print 'Accuracy:', accuracy_score(pred, labels_test),\
"Precision:", precision_score(pred, labels_test),\
"Recall", recall_score(pred, labels_test)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
