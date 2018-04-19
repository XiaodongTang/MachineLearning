import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')

from scipy.sparse.construct import hstack
from sklearn.model_selection import train_test_split
from sklearn.datasets.svmlight_format import load_svmlight_file
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.ranking import roc_auc_score
from sklearn.preprocessing.data import OneHotEncoder
import numpy as np

def gbdt_lr_train(libsvmFileName):
	X_all, y_all = load_svmlight_file(libsvmFileName)
	X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.3, random_state = 42)
	gbdt = GradientBoostingClassifier(n_estimators=50, max_depth=4, verbose=0,max_features=0.4)
	gbdt.fit(X_train, y_train)
	y_pred_gbdt = gbdt.predict_proba(X_test.toarray())[:, 1]
	gbdt_auc = roc_auc_score(y_test, y_pred_gbdt)
	score = gbdt.feature_importances_
	print(str(score))
	#print(str(gbdt.get_params(True)))
	#print(str(gbdt.train_score_))
	print('GBDT AUC: %.5f' % gbdt_auc)

	lr = LogisticRegression()
	lr.fit(X_train, y_train)

	y_pred_test = lr.predict_proba(X_test)[:, 1]
	lr_test_auc = roc_auc_score(y_test, y_pred_test)
	print('LR AUC: %.5f' % lr_test_auc)

	X_train_leaves = gbdt.apply(X_train)[:,:,0]
	X_test_leaves = gbdt.apply(X_test)[:,:,0]
	(train_rows, cols) = X_train_leaves.shape
	gbdtenc = OneHotEncoder()
	X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))

	lr = LogisticRegression()
	lr.fit(X_trans[:train_rows, :], y_train)
	y_pred_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
	gbdt_lr_auc1 = roc_auc_score(y_test, y_pred_gbdtlr1)
	print('GBDT + LR AUC: %.5f' % gbdt_lr_auc1)
if __name__ == '__main__':
	gbdt_lr_train('./feature5.txt')

