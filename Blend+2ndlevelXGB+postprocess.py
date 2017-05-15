#blending
# LightGBM
clfs = []
lgb_clfs = []
for p in lgb_BO_scores.head(10).iterrows():
    clfs.append(lgb.LGBMClassifier(n_estimators = 10000,
                                    learning_rate=0.01,
                                    max_bin=int(p[1].to_dict()['max_bins']),
                                    num_leaves=int(p[1].to_dict()['num_leaves']),
                                    min_child_weight=int(p[1].to_dict()['min_sum_hessian_in_leaf']),
                                    colsample_bytree=p[1].to_dict()['feature_fraction'],
                                    subsample=p[1].to_dict()['bagging_fraction'],
                                    subsample_freq=int(p[1].to_dict()['bagging_freq']),
                                    min_split_gain=p[1].to_dict()['min_gain_to_split'],
                                    seed=1234,
                                    nthread=-1)
                )

# XGBoost    
xgb_clfs = []
for p in BO_scores.head(10).iterrows():
    clfs.append(xgb.XGBClassifier(n_estimators = 10000,
                                  learning_rate=0.01,
                                  max_depth=int(p[1].to_dict()['max_depth']),
                                  min_child_weight=int(p[1].to_dict()['min_child_weight']),
                                  colsample_bytree=p[1].to_dict()['colsample_bytree'],
                                  subsample=p[1].to_dict()['subsample'],
                                  gamma=p[1].to_dict()['gamma'],
                                  seed=1234,
                                  nthread=-1))


def blend_model(clfs, train_x, train_y, test_x, num_class, blend_folds):
    num_class = 3
    blend_folds = 5

    skf = model_selection.StratifiedKFold(n_splits=blend_folds,random_state=1234)
    skf_ids = list(skf.split(train_x, train_y))


    train_blend_x = np.zeros((train_x.shape[0], len(clfs)*num_class))
    test_blend_x = np.zeros((test_x.shape[0], len(clfs)*num_class))
    blend_scores = np.zeros ((blend_folds,len(clfs)))

    print  ("Start blending.")
    for j, clf in enumerate(clfs):
        print ("Blending model",j+1, clf)
        test_blend_x_j = np.zeros((test_x.shape[0], num_class))
        for i, (train_ids, val_ids) in enumerate(skf_ids):
            print ("Model %d fold %d" %(j+1,i+1))
            train_x_fold = train_x[train_ids]
            train_y_fold = train_y[train_ids]
            val_x_fold = train_x[val_ids]
            val_y_fold = train_y[val_ids]
            # Set n_estimators to a large number for early_stopping
            clf.n_estimators = 10000000
            
            # Set evaluation metric
            if type(clf).__name__=='LGBMClassifier':
                metric = 'logloss' #LightGBM
            else:
                metric = 'mlogloss' #XGBoost            
            clf.fit(train_x_fold, train_y_fold,
                    eval_set=[(val_x_fold,val_y_fold)],
                    eval_metric=metric,
                    early_stopping_rounds=500,verbose=False)
            val_y_predict_fold = clf.predict_proba(val_x_fold)
            score = metrics.log_loss(val_y_fold,val_y_predict_fold)
            print ("LOGLOSS: ", score)
            print ("Best Iteration:", clf.best_iteration)
            blend_scores[i,j]=score
            train_blend_x[val_ids, j*num_class:j*num_class+num_class] = val_y_predict_fold
            test_blend_x_j = test_blend_x_j + clf.predict_proba(test_x)
        test_blend_x[:,j*num_class:j*num_class+num_class] = test_blend_x_j/blend_folds
        print ("Score for model %d is %f" % (j+1,np.mean(blend_scores[:,j])))
    return train_blend_x, test_blend_x, blend_scores

    train_blend_x, test_blend_x, blend_scores_x = blend_model(clfs, 
                                                        train_x, 
                                                        train_y, 
                                                        test_x, 
                                                        num_class=3, 
                                                        blend_folds=4)

    # MLP neural network as 2nd level (not as good as xgb as 2nd level on LB)
    from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.neural_network import MLPClassifier
def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):
##Grid Search for the best model
    model = model_selection.GridSearchCV(estimator  = est,
                                     param_grid = param_grid,
                                     scoring    = 'log_loss',
                                     verbose    = 10,
                                     n_jobs  = n_jobs,
                                     iid        = True,
                                     refit    = refit,
                                     cv      = cv)
    # Fit Grid Search Model
    model.fit(train_x, train_y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:", model.best_params_)
    print("Scores:", model.grid_scores_)
    return model


#### XGB as 2nd level
params = dict()
params['objective'] = 'multi:softprob'
params['num_class'] = 3
params['eta'] = 0.01
params['max_depth'] = 7
params['min_child_weight'] = 15
params['subsample'] = 1
params['colsample_bytree'] = 0.33
params['gamma'] = 0.35
params['seed']=1234

cv_results = xgb.cv(params, xgb.DMatrix(train_blend_x, label=train_y.reshape(train_x.shape[0],1)),
               num_boost_round=1000000, nfold=5,
       metrics={'mlogloss'},
       seed=1234,
       callbacks=[xgb.callback.early_stop(50)])

best_xgb_score = cv_results['test-mlogloss-mean'].min()
best_xgb_iteration = len(cv_results)
# cv score 0.5135

#post process make sure prepictions has the same distribution as train set
interest_levels = ['low', 'medium', 'high']

tau = {
    'low': 0.69195995, 
    'medium': 0.23108864,
    'high': 0.07695141, 
}

def correct(df):
    y = df[interest_levels].mean()
    a = [tau[k] / y[k]  for k in interest_levels]
    print(a)

    def f(p):
        for k in range(len(interest_levels)):
            p[k] *= a[k]
        return p / p.sum()

    df_correct = df.copy()
    df_correct[interest_levels] = df_correct[interest_levels].apply(f, axis=1)

    y = df_correct[interest_levels].mean()
    a = [tau[k] / y[k]  for k in interest_levels]
    print(a)

    return df_correct
# submit results
clf = xgb.XGBClassifier(learning_rate = 0.01
                  , n_estimators =best_xgb_iteration
                  , max_depth = 7
                  , min_child_weight = 15
                  , subsample = 1
                  , colsample_bytree = 0.33
                  , gamma = 0.35
                  , seed = 1234
                  , nthread = -1
                  )

clf.fit(train_blend_x, train_y)

preds = clf.predict_proba(test_blend_x)
sub_df3 = pd.DataFrame(preds,columns = ["low", "medium", "high"])
sub_df3["listing_id"] = test_data.listing_id.values
sub_df3.to_csv(r"..final.csv", index=False)
# final LB score 0.5130