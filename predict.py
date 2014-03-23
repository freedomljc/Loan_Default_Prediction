#! /usr/bin/env python

import re
import math
import collections
import numpy as np
import time
import operator
from scipy.io import mmread, mmwrite
from random import randint
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing as pp
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import  RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import ProbabilisticPCA, KernelPCA
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
import scipy.stats as stats
from sklearn import tree
from sklearn.feature_selection import f_regression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, f1_score
from sklearn.gaussian_process import GaussianProcess
import features

# working directory
dir = '.'
label_index = 770


# load train data
def load_train_fs():
    # In the validation process, the training data was randomly shuffled firstly. 
    # For the prediction process, there is no need to shuffle the dataset. 
    # Owing to out of memory problem, Gaussian process only use part of training data, the prediction of gaussian process
    # may be a little different from the model,which the training data was shuffled.
    train_fs = np.genfromtxt(open(dir + '/train_v2.csv','rb'), delimiter=',', skip_header=1)
    col_mean = stats.nanmean(train_fs, axis=0)
    inds = np.where(np.isnan(train_fs))
    train_fs[inds] = np.take(col_mean, inds[1])
    train_fs[np.isinf(train_fs)] = 0
    return train_fs


# load test data
def load_test_fs():
    test_fs = np.genfromtxt(open(dir + '/test_v2.csv','rb'), delimiter=',', skip_header = 1)
    col_mean = stats.nanmean(test_fs, axis=0)
    inds = np.where(np.isnan(test_fs))
    test_fs[inds] = np.take(col_mean, inds[1])
    test_fs[np.isinf(test_fs)] = 0
    return test_fs

# extract features from test data
def test_type(test_fs):
    x_Test = test_fs[:,range(1, label_index)]
    return x_Test

# extract features from train data
def train_type(train_fs):
    train_x = train_fs[:,range(1, label_index)]
    train_y= train_fs[:,-1]
    return train_x, train_y

# transform the loss to the binary form
def toLabels(train_y):
    labels = np.zeros(len(train_y))
    labels[train_y>0] = 1
    return labels

# generate the output file based to the predictions
def output_preds(preds):
    out_file = dir + '/output.csv'
    fs = open(out_file,'w')
    fs.write('id,loss\n')
    for i in range(len(preds)):
        if preds[i] > 100:
            preds[i] = 100
        elif preds[i] < 0:
            preds[i] = 0
        strs = str(i+105472) + ',' + str(np.float(preds[i]))
        fs.write(strs + '\n');
    fs.close()
    return

# get the top feature indexes by invoking f_regression 
def getTopFeatures(train_x, train_y, n_features=100):
    f_val, p_val = f_regression(train_x,train_y)
    f_val_dict = {}
    p_val_dict = {}
    for i in range(len(f_val)):
        if math.isnan(f_val[i]):
            f_val[i] = 0.0
        f_val_dict[i] = f_val[i]
        if math.isnan(p_val[i]):
            p_val[i] = 0.0
        p_val_dict[i] = p_val[i]
    
    sorted_f = sorted(f_val_dict.iteritems(), key=operator.itemgetter(1),reverse=True)
    sorted_p = sorted(p_val_dict.iteritems(), key=operator.itemgetter(1),reverse=True)
    
    feature_indexs = []
    for i in range(0,n_features):
        feature_indexs.append(sorted_f[i][0])
    
    return feature_indexs

# generate the new data, based on which features are generated, and used
def get_data(train_x, feature_indexs, feature_minus_pair_list=[], feature_plus_pair_list=[],
            feature_mul_pair_list=[], feature_divide_pair_list = [], feature_pair_sub_mul_list=[],
            feature_pair_plus_mul_list = [],feature_pair_sub_divide_list = [], feature_minus2_pair_list = [],feature_mul2_pair_list=[], 
            feature_sub_square_pair_list=[], feature_square_sub_pair_list=[],feature_square_plus_pair_list=[]):
    sub_train_x = train_x[:,feature_indexs]
    for i in range(len(feature_minus_pair_list)):
        ind_i = feature_minus_pair_list[i][0]
        ind_j = feature_minus_pair_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i]-train_x[:,ind_j]))
    
    for i in range(len(feature_plus_pair_list)):
        ind_i = feature_plus_pair_list[i][0]
        ind_j = feature_plus_pair_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] + train_x[:,ind_j]))
    
    for i in range(len(feature_mul_pair_list)):
        ind_i = feature_mul_pair_list[i][0]
        ind_j = feature_mul_pair_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] * train_x[:,ind_j]))
    
    for i in range(len(feature_divide_pair_list)):
        ind_i = feature_divide_pair_list[i][0]
        ind_j = feature_divide_pair_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] / train_x[:,ind_j]))
    
    for i in range(len(feature_pair_sub_mul_list)):
        ind_i = feature_pair_sub_mul_list[i][0]
        ind_j = feature_pair_sub_mul_list[i][1]
        ind_k = feature_pair_sub_mul_list[i][2]
        sub_train_x = np.column_stack((sub_train_x, (train_x[:,ind_i]-train_x[:,ind_j]) * train_x[:,ind_k]))
        
    return sub_train_x

# use gbm classifier to predict whether the loan defaults or not
def gbc_classify(train_x, train_y):
    feature_indexs = getTopFeatures(train_x, train_y)
    sub_x_Train = get_data(train_x, feature_indexs[:16], features.feature_pair_sub_list
                ,features.feature_pair_plus_list, features.feature_pair_mul_list, features.feature_pair_divide_list[:20],
                features.feature_pair_sub_mul_list[:20])
    labels = toLabels(train_y)
    gbc = GradientBoostingClassifier(n_estimators=3000, max_depth=8)
    gbc.fit(sub_x_Train, labels)
    return gbc

# use svm to predict the loss, based on the result of gbm classifier
def gbc_svr_predict_part(gbc, train_x, train_y, test_x, feature_pair_sub_list, 
                     feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list, 
                     feature_pair_sub_mul_list, feature_pair_sub_list_sf, feature_pair_plus_list2):
    feature_indexs = getTopFeatures(train_x, train_y)
    sub_x_Train = get_data(train_x, feature_indexs[:16], feature_pair_sub_list
                ,feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list[:20], feature_pair_sub_mul_list[:20])
    sub_x_Test = get_data(test_x, feature_indexs[:16], feature_pair_sub_list
                ,feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list[:20], feature_pair_sub_mul_list[:20])
    pred_labels = gbc.predict(sub_x_Test)
    
    pred_probs = gbc.predict_proba(sub_x_Test)[:,1]
    
    ind_test = np.where(pred_probs>0.55)[0]
    
    ind_train = np.where(train_y > 0)[0]
    ind_train0 = np.where(train_y == 0)[0]
    
    preds_all = np.zeros([len(sub_x_Test)])
    
    flag = (sub_x_Test[:,16] >= 1) 
    ind_tmp0 = np.where(flag)[0]
    
    ind_tmp = np.where(~flag)[0]
    
    sub_x_Train = get_data(train_x, feature_indexs[:100], feature_pair_sub_list_sf
        ,feature_pair_plus_list2[:100], feature_pair_mul_list[:40], feature_pair_divide_list, feature_pair_sub_mul_list)
    sub_x_Test = get_data(test_x, feature_indexs[:100], feature_pair_sub_list_sf
        ,feature_pair_plus_list2[:100], feature_pair_mul_list[:40], feature_pair_divide_list, feature_pair_sub_mul_list)
    sub_x_Train[:,101] = np.log(1-sub_x_Train[:,101])
    sub_x_Test[ind_tmp,101] = np.log(1-sub_x_Test[ind_tmp,101])
    scaler = pp.StandardScaler()
    scaler.fit(sub_x_Train)
    sub_x_Train = scaler.transform(sub_x_Train)
    sub_x_Test[ind_tmp] = scaler.transform(sub_x_Test[ind_tmp]) 
    
    svr = SVR(C=16, kernel='rbf', gamma =  0.000122)
    
    svr.fit(sub_x_Train[ind_train], np.log(train_y[ind_train]))
    preds = svr.predict(sub_x_Test[ind_test])
    preds_all[ind_test] = np.power(np.e, preds)
    preds_all[ind_tmp0] = 0
    return preds_all

# use gbm regression to predict the loss, based on the result of gbm classifier
def gbc_gbr_predict_part(gbc,  train_x, train_y, test_x, feature_pair_sub_list, 
                     feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list, 
                     feature_pair_sub_mul_list, feature_pair_sub_list2):
    feature_indexs = getTopFeatures(train_x, train_y)
    sub_x_Train = get_data(train_x, feature_indexs[:16], feature_pair_sub_list
                ,feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list[:20],feature_pair_sub_mul_list[:20])
    sub_x_Test = get_data(test_x, feature_indexs[:16], feature_pair_sub_list
                ,feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list[:20], feature_pair_sub_mul_list[:20])
    pred_labels = gbc.predict(sub_x_Test)
    
    pred_probs = gbc.predict_proba(sub_x_Test)[:,1]
    
    ind_test = np.where(pred_probs>0.55)[0]
    
    ind_train = np.where(train_y > 0)[0]
    ind_train0 = np.where(train_y == 0)[0]
    
    preds_all = np.zeros([len(sub_x_Test)])
    
    flag = (sub_x_Test[:,16] >= 1) 
    ind_tmp0 = np.where(flag)[0]
    
    ind_tmp = np.where(~flag)[0]
    
    sub_x_Train = get_data(train_x, feature_indexs[:16], feature_pair_sub_list2[:70]
                        ,feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list, feature_pair_sub_mul_list)
    sub_x_Test = get_data(test_x, feature_indexs[:16], feature_pair_sub_list2[:70]
                        ,feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list, feature_pair_sub_mul_list)
    
    scaler = pp.StandardScaler()
    scaler.fit(sub_x_Train)
    sub_x_Train = scaler.transform(sub_x_Train)
    sub_x_Test[ind_tmp] = scaler.transform(sub_x_Test[ind_tmp]) 
    
    gbr1000 = GradientBoostingRegressor(n_estimators=1300, max_depth=4, subsample=0.5, learning_rate=0.05)
    
    gbr1000.fit(sub_x_Train[ind_train], np.log(train_y[ind_train]))
    preds = gbr1000.predict(sub_x_Test[ind_test])
    preds_all[ind_test] = np.power(np.e, preds)
    preds_all[ind_tmp0] = 0
    return preds_all


# predict the loss based on the Gaussian process regressor, which has been trained
def gp_predict(clf, x_Test):
    size = len(x_Test)
    part_size = 3000
    cnt = (size-1) / part_size + 1
    preds = []
    for i in range(cnt):
        if i < cnt - 1:
            pred_part = clf.predict(x_Test[i*part_size: (i+1) * part_size])
        else:
            pred_part = clf.predict(x_Test[i*part_size: size])
        preds.extend(pred_part)
    return np.power(np.e,preds)

# train the gaussian process regressor
def gbc_gp_predict_part(sub_x_Train, train_y, sub_x_Test_part):
    #Owing to out of memory, the model was trained by part of training data
    #Attention, this part was trained on the ram of more than 96G
    sub_x_Train[:,16] = np.log(1-sub_x_Train[:,16])
    scaler = pp.StandardScaler()
    scaler.fit(sub_x_Train)
    sub_x_Train = scaler.transform(sub_x_Train)
    ind_train = np.where(train_y>0)[0]
    part_size= int(0.7 * len(ind_train))
    gp = GaussianProcess(theta0=1e-3, thetaL=1e-5, thetaU=10, corr= 'absolute_exponential')
    gp.fit(sub_x_Train[ind_train[:part_size]], np.log(train_y[ind_train[:part_size]]))
    flag = (sub_x_Test_part[:,16] >= 1)
    ind_tmp0 = np.where(flag)[0]
    ind_tmp = np.where(~flag)[0]
    sub_x_Test_part[ind_tmp,16] = np.log(1-sub_x_Test_part[ind_tmp,16])
    sub_x_Test_part[ind_tmp] = scaler.transform(sub_x_Test_part[ind_tmp]) 
    gp_preds_tmp = gp_predict(gp, sub_x_Test_part[ind_tmp])
    gp_preds = np.zeros(len(sub_x_Test_part))
    gp_preds[ind_tmp] = gp_preds_tmp
    return gp_preds

# use gbm classifier to predict whether the loan defaults or not, then invoke the function gbc_gp_predict_part
def gbc_gp_predict(train_x, train_y, test_x):
    feature_indexs = getTopFeatures(train_x, train_y)
    sub_x_Train = get_data(train_x, feature_indexs[:16], features.feature_pair_sub_list
            ,features.feature_pair_plus_list, features.feature_pair_mul_list, features.feature_pair_divide_list[:20])
    sub_x_Test = get_data(test_x, feature_indexs[:16], features.feature_pair_sub_list
            ,features.feature_pair_plus_list, features.feature_pair_mul_list, features.feature_pair_divide_list[:20])
    labels = toLabels(train_y)
    gbc = GradientBoostingClassifier(n_estimators=3000, max_depth=9)
    gbc.fit(sub_x_Train, labels)
    pred_probs = gbc.predict_proba(sub_x_Test)[:,1]
    ind_test = np.where(pred_probs>0.55)[0]
    gp_preds_part = gbc_gp_predict_part(sub_x_Train, train_y, sub_x_Test[ind_test])
    gp_preds = np.zeros(len(test_x))
    gp_preds[ind_test] = gp_preds_part
    return gp_preds


# invoke the function gbc_svr_predict_part
def gbc_svr_predict(gbc, train_x, train_y, test_x):
    svr_preds = gbc_svr_predict_part(gbc, train_x, train_y, test_x, features.feature_pair_sub_list, features.feature_pair_plus_list, 
                                     features.feature_pair_mul_list, features.feature_pair_divide_list,
                                     features.feature_pair_sub_mul_list, features.feature_pair_sub_list_sf, 
                                     features.feature_pair_plus_list2)
    return svr_preds

# invoke the function gbc_gbr_predict_part
def gbc_gbr_predict(gbc, train_x, train_y, test_x):
    gbr_preds = gbc_gbr_predict_part(gbc,  train_x, train_y, test_x, features.feature_pair_sub_list, 
                                     features.feature_pair_plus_list, features.feature_pair_mul_list, 
                                     features.feature_pair_divide_list, features.feature_pair_sub_mul_list, 
                                     features.feature_pair_sub_list2)
    return gbr_preds

# the main function
if __name__ == '__main__':
    train_fs = load_train_fs()
    test_fs = load_test_fs()
    train_x, train_y = train_type(train_fs)
    test_x = test_type(test_fs)
    gbc = gbc_classify(train_x, train_y)
    svr_preds = gbc_svr_predict(gbc, train_x, train_y, test_x)
    gbr_preds = gbc_gbr_predict(gbc, train_x, train_y, test_x)
    gp_preds = gbc_gp_predict(train_x, train_y, test_x)
    preds_all = svr_preds * 0.5 + gp_preds * 0.2 + gbr_preds * 0.3
    output_preds(preds_all)
    

