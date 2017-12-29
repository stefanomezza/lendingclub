from LendingClub import LendingClub
from sklearn.linear_model import LogisticRegression
import sys
LC=LendingClub(sys.argv[1],parameters={'max_features':[1,5,10,None],'min_weight_fraction_leaf':[0.1,0.3,0.5],'max_depth':[1,5,10]})
#LC=LendingClub(sys.argv[1],regressor=LogisticRegression(),parameters={'C':[100.0,10.0,1.0,0.5,0.3,0.1,0.01,1e-8]})
#LC=LendingClub(sys.argv[1],regressor=SVR(),parameters={'C':[0.01,0.1,1.0,10.0],'gamma':[1e-10,1e-8,1e-3,0.1]})
LC.train_regressor()
LC.test_regressor()
metrics=LC.evaluate_regressor()
for m in metrics:
	print m,":",metrics[m]
LC.show_ROC_curve()
