# creating data
full_vars = num_vars + date_num_vars\
+ additional_num_vars + interactive_num_vars + LE_vars + hcc_vars + num_cat_vars + manager_variable + dist_var #+ intr_vars
#,st_addr_sparse 
train_x = sparse.hstack([full_data[full_vars],feature_sparse]).tocsr()[:train_size]
train_y = full_data['target'][:train_size].values

test_x = sparse.hstack([full_data[full_vars], feature_sparse]).tocsr()[train_size:]
test_y = full_data['target'][train_size:].values

from bayes_opt import BayesianOptimization

# 10 tuned xgb models
xgtrain = xgb.DMatrix(train_x, label=train_y.reshape(train_x.shape[0],1))

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma):
    params = dict()
    params['objective'] = 'multi:softprob'
    params['num_class'] = 3
    params['eta'] = 0.1
    params['max_depth'] = int(max_depth )   
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = colsample_bytree
    params['subsample'] = subsample
    params['gamma'] = gamma
    params['verbose_eval'] = True    


    cv_result = xgb.cv(params, xgtrain,
                       num_boost_round=100000,
                       nfold=5,
                       metrics={'mlogloss'},
                       seed=1234,
                       callbacks=[xgb.callback.early_stop(50)])

    return -cv_result['test-mlogloss-mean'].min()


xgb_BO = BayesianOptimization(xgb_evaluate, 
                             {'max_depth': (3, 9),
                              'min_child_weight': (0, 100),
                              'colsample_bytree': (0.1, 0.8),
                              'subsample': (0.7, 1),
                              'gamma': (0, 2)
                             }
                            )

xgb_BO.maximize(init_points=6, n_iter=120)

#xbg results
BO_scores = pd.DataFrame(xgb_BO.res['all']['params'])
BO_scores['score'] = pd.DataFrame(xgb_BO.res['all']['values'])
BO_scores = BO_scores.sort_values(by='score',ascending=False)


#lightGBM models
import lightgbm as lgb
lgb_train = lgb.Dataset(train_x, train_y)

def lgb_evaluate(max_bins,
                 num_leaves,
                 min_sum_hessian_in_leaf,
                 min_gain_to_split,
                 feature_fraction,
                 bagging_fraction,
                 bagging_freq
                 ):
    params = dict()
    params['objective'] = 'multiclass'
    params['num_class'] = 3
    params['learning_rate'] = 0.1
    params['max_bins'] = int(max_bins)   
    params['num_leaves'] = int(num_leaves)    
    params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
    params['min_gain_to_split'] = int(min_gain_to_split)    
    params['feature_fraction'] = feature_fraction
    params['bagging_fraction'] = bagging_fraction
    params['bagging_freq'] = int(bagging_freq)


    cv_results = lgb.cv(params,
                    lgb_train,
                    num_boost_round=100000,
                    nfold=5,
                    early_stopping_rounds=100,
                    metrics='multi_logloss',
                    verbose_eval=False
                   )

    return -pd.DataFrame(cv_results)['multi_logloss-mean'].min()


lgb_BO = BayesianOptimization(lgb_evaluate, 
                             {'max_bins': (127, 1023),
                              'num_leaves': (15, 512),
                              'min_sum_hessian_in_leaf': (1, 100),
                              'min_gain_to_split': (0,2),
                              'feature_fraction': (0.2, 0.8),
                              'bagging_fraction': (0.7, 1),
                              'bagging_freq': (1, 5)
                             }
                            )

lgb_BO.maximize(init_points=7, n_iter=150)

#lgb tuning tuning cv_results
lgb_BO_scores = pd.DataFrame(lgb_BO.res['all']['params'])
lgb_BO_scores['score'] = pd.DataFrame(lgb_BO.res['all']['values'])
lgb_BO_scores = lgb_BO_scores.sort_values(by='score',ascending=False)