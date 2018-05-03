import pandas as pd
import numpy as np
from sklearn import datasets,model_selection,tree,ensemble

def quality(classifier,data,target):
	return model_selection.cross_val_score(classifier,data,target,cv=5).mean()

dataset=datasets.load_digits()
data,target=dataset.data,dataset.target

tree_classifier=tree.DecisionTreeClassifier().fit(data,target)

print(quality(tree_classifier,data,target)) #answer № 1

bag_classifier=ensemble.BaggingClassifier(n_estimators=100).fit(data,target)

print(quality(bag_classifier,data,target)) #answer № 2

bag_square_classifier=ensemble.BaggingClassifier(n_estimators=100,max_features=int(np.sqrt(len(data)))).fit(data,target)

print(quality(bag_square_classifier,data,target)) #answer № 3

bag_square_rand_classifier=ensemble.BaggingClassifier(n_estimators=100,base_estimator=tree.DecisionTreeClassifier(max_features=int(np.sqrt(len(data))))).fit(data,target)

print(quality(bag_square_rand_classifier,data,target)) #answer № 4

print('2,3,4,7') #answer № 5 maybe?