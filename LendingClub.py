import csv
from sklearn.svm import SVR
from random import shuffle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error,roc_curve,auc,confusion_matrix,make_scorer,roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import os.path

class LendingClub:
	def __init__(self,in_file,regressor=GradientBoostingRegressor(),parameters={}):
		"""Constructor of the LendingClub class
    		Args:
		  in_file: a CSV file containing the LendingClub data
		  regressor: an sklearn regressor (default: GradientBoostingRegressor)
		  parameters: a dictionary of parameters to tune for the classifier (default:None)

    		Raises:
        		AssertionError: in_file is not a valid CSV file
    		"""
		self.regressor=GridSearchCV(regressor,parameters,cv=3,scoring=make_scorer(roc_auc_score))
		self.chi2=SelectKBest(chi2,k=15)
		self.vectorizer=DictVectorizer()
		self.predictions=[] #this will store predictions from the classifier
		try:
			assert(os.path.isfile(in_file))
		except AssertionError:
			print "Error: ",in_file," is not a valid CSV file"
			exit(1)
		data=self.readfile(in_file) #read CSV file and build a data structure
		shuffle(data) #shuffle data to distribute labels
		size=int(len(data)*0.7)
		self.train_set=data[:size] #split data in train-test sets
		self.test_set=data[size:]
		self.oversample() #add oversampling for the "Uncreditworthy" class

	def oversample(self,c=0,factor=2):
		"""Add examples from the c class
    		Args:
		  c: the class to oversample (default=0)
		  factor: defines how many times examples must be oversampled. (default=2)
		Note:
		  data are shuffled again after the oversampling to distribute labels
		"""
		self.train_set=self.train_set+factor*[t for t in self.train_set if t[1]==c]
		shuffle(self.train_set)

	def readfile(self,in_file):
		"""Add examples from the c class
    		Args:
		  in_file: directory of the input CSV file to open
		Returns:
		  An array of tuples (D_i,y_i), where D_i is a dictionary mapping
		  features from example i to their values and y_i is the output
		  label for example i.
		"""
		data=[] #this will contain the loaded data
		keys=[] #this will contain the list of available features in the CSV
		with open(in_file, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for (i,row) in enumerate(reader):
				if i==0: #first row: load feature keys
					for k in row:
						field=k.replace(" ","_").lower()
						keys.append(field)
				else: #each row is data from a lend
					lend={}
					class_label=None
					for i,field in enumerate(row):
						if keys[i]=="class": #class is loaded separately and mapped to [0,1]
							if field=="Creditworthy":
								class_label=1
							else:
								class_label=0
						else: #load features from this lend 
							lend[keys[i]]=field
					data.append((lend,class_label)) #add the lend data to the data list
		return data

	def extract_numeric(self,value):
		"""Given the value for a feature, it tries to map it to a numeric value
    		Args:
		  value: the value of the input feature
		Returns:
		  A numeric representation of the feature value or None if such representation
		  doesn't exist
		"""
		numeric_value=None
		for w in value.split(" "):
			try:
				numeric_value=int(float(w)) #if the string contains a numeric, it is stored as its value
			except:
				if w is None: #None maps to zero
					numeric_value=0
				pass
		if numeric_value==None: #try to map string representations of integers to integers
			numeric_value={"NA":0,"One":1,"Two":2,"Three":3,"Four":4,"Five":5,"Six":6,"None":0}.get(value,None)
		return numeric_value
			
	def field_vectorize(self,field_row):
		"""Given a row of features, it maps their value to a numeric or integer
    		Args:
		  field_row: a row of features from the parsed CSV
		Returns:
		  A row with numeric values mapped to a numeric field and other values mapped to a string representation
		"""
		mapped_row={}
		for field in field_row:
			mapped_value=self.extract_numeric(field_row[field])
			if mapped_value==None: #not a numeric, keep the string representation
				mapped_value=field_row[field]
			mapped_row[field]=mapped_value
		return mapped_row

	def train_regressor(self):
		"""train a regressor for the loaded training data"""
		vectorized_data=self.vectorizer.fit_transform([(self.field_vectorize(X)) for X,y in self.train_set]) 
		labels=[y for X,y in self.train_set]
		vectorized_data=self.chi2.fit_transform(vectorized_data,labels) #vectorize using a DictVectorizer. chi2 used to choose features
		self.regressor.fit(vectorized_data.toarray(),labels)	
			
	def test_regressor(self):
		"""test the trained regressor on the loaded test data
		Raises:
			AssertionError: the regressor hasn't been trained yet
		"""
		try:
			vectorized_data=self.vectorizer.transform([(self.field_vectorize(X)) for X,y in self.test_set])
		except AttributeError:
			print "Error: testing without training: run <LendingClub object>.train_regressor() before running the test_regressor method"
			exit(1)
		vectorized_data=self.chi2.transform(vectorized_data)
		self.predictions=self.regressor.predict(vectorized_data.toarray())

	def evaluate_regressor(self):
		"""evaluate the output of the regressor on the loaded test data
		Raises:
			AssertionError: the regressor hasn't been tested yet
		"""
		try:
			assert(len(self.predictions)>0)
		except AssertionError:
			print "Error: evaluating without testing: run <LendingClub object>.test_regressor() before running the evaluate_regressor method"
			exit(1)
		y_true=[y for X,y in self.test_set]
		false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, self.predictions)
		roc_auc = auc(false_positive_rate, true_positive_rate)
		return {"MSE":mean_squared_error(y_true,self.predictions),"Confusion Matrix":confusion_matrix(y_true,[int(round(y)) for y in self.predictions]),"Ghini coefficient":(2*roc_auc-1)}

	def show_ROC_curve(self):
		"""draw the ROC curve for the current prediction
		Raises:
			AssertionError: the regressor hasn't been tested yet
		"""
		try:
			assert(len(self.predictions)>0)
		except AssertionError:
			print "Error: evaluating without testing: run <LendingClub object>.test_regressor() before running the show_ROC_curve method"
			exit(1)
		y_true=[y for X,y in self.test_set]
		false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, self.predictions)
		roc_auc = auc(false_positive_rate, true_positive_rate)
		plt.title('Receiver Operating Characteristic')
		plt.plot(false_positive_rate, true_positive_rate, 'b',
		label='AUC = %0.2f'% roc_auc)
		plt.legend(loc='lower right')
		plt.plot([0,1],[0,1],'r--')
		plt.xlim([-0.1,1.2])
		plt.ylim([-0.1,1.2])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()

