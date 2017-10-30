import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
# This class is for the preprocessing, take csv path as input, and aim to return a pandas data frame for trainning
class PreProcessing:

  # The constructor takes a pandas dataframe as input and save it to self.df
  def __init__(self, csvpath):
    self.df = pd.read_csv(csvpath)

  # This method have deal with missing data before merge or drop
  def MissingData(self):
    #self.df = self.df.replace(-1, np.NaN)
    #df[""].fillna(df[""].mean(), inplace=True)
    #df[""].fillna(df[""].median(), inplace=True)
    #df[""].fillna(df[""].value_counts().idxmax(), inplace=True)
    #mean_imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    #mdan_imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    #mfrq_imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    return

  # This method drop or merge variables in dataframe accroding to corr map
  def CorrMergeDrop(self):
    self.df['ps_ind_06070809_bin'] = self.df.apply(
      lambda x: 1 if x['ps_ind_06_bin'] == 1 
                  else 
                  (2 if x['ps_ind_07_bin'] == 1 
                     else 
                     ( 3 if x['ps_ind_08_bin'] == 1 
                         else 
                         (4 if x['ps_ind_09_bin'] == 1 else 5)
                     )
                  ), axis = 1)
    self.df.drop(['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin'], axis = 1, inplace = True)

    self.df['ps_ind_161718_bin'] = self.df.apply(lambda x: 1 if x['ps_ind_16_bin'] == 1 
                                                             else (2 if x['ps_ind_17_bin'] == 1 else 3), axis = 1)
    self.df.drop(['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'], axis = 1, inplace = True)
 
    # drop this variable from preprocessing study, top 3 missing data, and not important at all
    #self.df.drop(['ps_car_05_cat'], axis = 1, inplace = True)
   
    #self.df['ps_car_13'] = (self.df['ps_car_13']*self.df['ps_car_13']*48400).round(0)
    #self.df['ps_car_12'] = (self.df['ps_car_12']*self.df['ps_car_12']).round(4) * 10000
    return

  # this method pack all previous preprocessing all together and return the data frame
  def FinalFrameforTrainning(self):
    self.MissingData()
    #self.CorrMergeDrop()
    return self.df

from sklearn.metrics import roc_auc_score
# this is a simple function to calculate gini score, for cross validation
def GiniScore(y_actual, y_pred):
  return 2*roc_auc_score(y_actual, y_pred)-1

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
# this class define various trainning methods, also cross validation to tune the hyper parameters
class Train:

  # The constructor takes a pandas dataframe as input and save it to self.df
  def __init__(self, df_train):
    self.df = df_train

  # This method evaluate the preprocessing with rf regression model, calculate the mean absolute error. Be careful! this method can only used in train set!
  def PreprocessingScore(self):
    X_train, X_test, y_train, y_test = train_test_split(
                                                        self.df.drop(['id', 'target'],axis=1),
                                                        self.df.target,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0
                                                       )

    sc_mod = RandomForestClassifier(n_estimators=500, max_features='auto', criterion='gini', max_depth=10, min_samples_leaf=10, min_samples_split=15, n_jobs=-1, random_state=0)
    sc_mod.fit(X_train, y_train)
    ysc_pred = sc_mod.predict_proba(X_test)[:,1]
    #print (ysc_pred)
    #print (y_test)
    gini = 2*roc_auc_score(y_test, ysc_pred)-1
    return gini

  # A simple train of random forest model with scikit learn
  def TrainSKLearnRandomForest(self):
    rf = RandomForestClassifier(n_estimators=500, max_features='auto', criterion='gini', max_depth=10, min_samples_leaf=10, min_samples_split=15, n_jobs=-1, random_state=0)
    rf.fit(self.df.drop(['id', 'target'],axis=1), self.df.target)
    #features = self.df.drop(['id', 'target'],axis=1).columns.values
    return rf

  # XGBoost
  def TrainXGBoost(self):
    xgb = XGBClassifier(    
                        n_estimators=350,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=.77,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                       )
    fit_xgb = xgb.fit(self.df.drop(['id', 'target'],axis=1), self.df.target)
    return fit_xgb
  
# this class makes prediction, with various methods
class Prediction:

  # The constructor takes model and test data for 
  def __init__(self, df_test, thismodel):
    self.df = df_test
    self.model = thismodel

  # prediction with simple random forest model
  def PredSKLearnRandomForest(self):
    pred = self.model.predict_proba(self.df.drop(['id'],axis=1))
    return pred[:,1]

  # prediction with xgb
  def PredXGBoost(self):
    pred = self.model.predict_proba(self.df.drop(['id'],axis=1))[:,1]
    return pred

# Create submission file
class CreateSubmitCSV:

  # The constructor ypred and test dataframe as input 
  def __init__(self, test_id, pred_target):
    self.testid = test_id
    self.predtarget = pred_target

  def Create(self):
    # Create submission file
    sub = pd.DataFrame()
    sub['id'] = self.testid
    sub['target'] = self.predtarget
    sub.to_csv('this_submit.csv', index=False)

if __name__ == '__main__':
  #'''
  preprocessing = PreProcessing('data/train.csv')
  train_p = preprocessing.FinalFrameforTrainning()
  print ("done with trainning set preprocessing!")
  #train_p.to_csv('train_p.csv', index = False)
  #preprocessing = PreProcessing('data/test.csv')
  #test_p = preprocessing.FinalFrameforTrainning()
  #print ("done with test set preprocessing!")
  #test_p.to_csv('test_p.csv', index = False)
  #train_p = pd.read_csv('train_p.csv')
  #test_p = pd.read_csv('test_p.csv')
  
  train = Train(train_p)
  print (train.PreprocessingScore())
  '''
  thisrf = train.TrainSKLearnRandomForest()
  #thisxgb = train.TrainXGBoost()
  print ("done with model trainning!")
  
  prediction = Prediction(test_p, thisrf)
  ypred = prediction.PredSKLearnRandomForest()
  #prediction = Prediction(test_p, thisxgb)
  #ypred = prediction.PredXGBoost()
  print ("done with prediction!")
  print (ypred) 

  outcsv = CreateSubmitCSV(test_p['id'], ypred)
  outcsv.Create()
  print ("done with csv output!")
  '''

