import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeRegressor

from MLE import MultiLabelEncoder

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
norm = Normalizer('l2')

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
pd.set_option('display.max_rows', 1000)

trainingFile = "./data/train.csv"
testFile = "./data/test.csv"

trainDf = pd.read_csv(trainingFile, header=0)
testDf = pd.read_csv(testFile, header=0)

trainDf['MoSold'] = trainDf['MoSold'].apply(str)
testDf['MoSold'] = testDf['MoSold'].apply(str)

# trainDf['OverallQual'] = trainDf['OverallQual'].apply(np.log)
# #trainDf['GrLivArea'] = trainDf['GrLivArea'].pow(3) #nope
# trainDf['GarageCars'] = trainDf['GarageCars'].pow(2)
# trainDf['GarageArea'] = trainDf['GarageArea'].pow(2)
#
# testDf['OverallQual'] = testDf['OverallQual'].apply(np.log)
# #testDf['GrLivArea'] = testDf['GrLivArea'].pow(3)
# testDf['GarageCars'] = testDf['GarageCars'].pow(2)
# testDf['GarageArea'] = testDf['GarageArea'].pow(2)

target = 'SalePrice'

Y = trainDf[target]


def prepare_data():
    training = trainDf.drop(['Id', target], axis=1)
    testing = testDf.drop(['Id'], axis=1)

    str_cols = []
    str_cols_idx = []

    pos = 0
    for c in training.columns:
        if c != target:
            if training[c].dtype == np.object:
                str_cols.append(c)
                str_cols_idx.append(pos)
            pos = pos + 1

    print("Number of string columns %d " % len(str_cols))

    for c in str_cols:
        training[c] = training[c].fillna("$NULL")
        testing[c] = testing[c].fillna("$NULL")

    print(training.dtypes)
    print(training.head(10))

    enc = MultiLabelEncoder(input_cols=np.array(str_cols_idx))

    t_pipe = Pipeline(steps=[
        ('catencode', enc),
        ('null_handler', imp)])

    fit_pipeline = t_pipe.fit(pd.concat([training, testing], axis=0))

    transformed = fit_pipeline.transform(training)

    transformed_test = fit_pipeline.transform(testing)

    print("T_TRAIN: ", transformed)
    print("T_TEST: ", transformed_test)

    return (pd.DataFrame(data=transformed, columns=training.columns),
            pd.DataFrame(data=transformed_test, columns=testing.columns))


def correlations(t_df):
    correlations = {}
    features = t_df.columns

    for f in features:
        if f != target:
            x1 = t_df[f]
            key = f + ' vs ' + target
            correlations[key] = pearsonr(x1, Y)[0]

    data_correlations = pd.DataFrame(correlations, index=['Value']).T
    sorted_c = data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]

    pd.set_option('display.max_rows', None)
    print(sorted_c)


def cv(df, pipeline):
    iter_rmsle = []
    iteration = 0
    kf = KFold(n_splits=10, random_state=0)
    for train_idx, test_idx in kf.split(df):
        print("KFold iteration ", iteration)
        x_train, x_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        model = pipeline.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        mse = mean_squared_log_error(y_test, y_predict)
        rmsle = math.sqrt(mse)
        print(rmsle)
        iter_rmsle.append(rmsle)
        iteration += 1

    return np.mean(iter_rmsle)


def select_pipeline(t_df, make_pipelines):
    c_df = t_df.copy()
    rmsles = []
    pipelines = []

    for pipeline in make_pipelines():
        mean = cv(c_df, pipeline)

        print("Mean RMSLE: ", mean)
        rmsles.append(mean)
        pipelines.append(pipeline)

    min_index = np.argmin(rmsles)
    print('Min RMSLE: ', np.min(rmsles))
    print('Min RMSLE index: ', min_index)

    best_pipeline = pipelines[min_index]
    print('Best pipeline', best_pipeline)

    best_model = best_pipeline.fit(c_df, Y)

    print("RMSLES : ", rmsles)

    return (best_model, rmsles)


def decision_tree_regressor():
    pipelines = []
    est = DecisionTreeRegressor(criterion='mse', max_depth=7, max_features=None,
                                max_leaf_nodes=None, min_impurity_decrease=0.0,
                                min_impurity_split=None, min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                presort=False, random_state=0, splitter='best')
    pipelines.append(Pipeline(steps=[('lr', est)]))
    return pipelines


def xgb_regressor():
    pipelines = []
    for l in [0, 0.5, 0.7, 1.0, 1.5, 2]:
        est = xgboost.XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.7, gamma=0,
                                   learning_rate=0.05, max_delta_step=0, max_depth=5,
                                   min_child_weight=4, missing=None, n_estimators=500, nthread=-1,
                                   objective='reg:linear', reg_alpha=0, reg_lambda=l,
                                   scale_pos_weight=1, seed=0, silent=True, subsample=0.75)

        pipelines.append(Pipeline(steps=[('lr', est)]))

    return pipelines


def ridge():
    pipelines = []
    for l in [0, 0.5, 0.7, 1.0, 1.5, 2]:
        est = Ridge(alpha=l, copy_X=False, fit_intercept=True, max_iter=None,
                    normalize=True, random_state=0, solver='auto', tol=0.001)
        pipelines.append(Pipeline(steps=[('lr', est)]))
    return pipelines


def linear():
    pipelines = []
    est = LinearRegression(normalize=True)
    pipelines.append(Pipeline(steps=[('lr', est)]))
    return pipelines


def lasso():
    pipelines = []
    for l in [0, 0.5, 0.7, 1.0, 1.5, 2]:
        est = Lasso(alpha=l, copy_X=True, fit_intercept=True, max_iter=1000,
                    normalize=False, positive=False, precompute=False, random_state=0,
                    selection='cyclic', tol=0.0001, warm_start=False)
        pipelines.append(Pipeline(steps=[('lr', est)]))
    return pipelines


def predict(model, testing):
    sp_id = testDf['Id']

    pred = model.predict(testing)

    result = pd.DataFrame({'Id': sp_id, 'SalePrice': pred}, index=None)

    print(result.head(10))
    result.to_csv('./submission.csv', index=False)
    print("Submission file created")


def stacking(training,
             y,
             test,
             pipelines):

    train, valid, y_train, y_valid = train_test_split(training, y, test_size=0.5)

    meta_validation = []
    meta_test = []

    for alg in pipelines:
        t_cpy = train.copy()
        alg_model = alg.fit(t_cpy, y_train)

        valid_pred = alg_model.predict(valid)
        test_pred = alg_model.predict(test)

        meta_validation.append(valid_pred)
        meta_test.append(test_pred)

    stacked_valid = np.column_stack(tuple(meta_validation))

    stacked_test = np.column_stack(tuple(meta_test))

    print("Stacked valid data: ", stacked_valid)

    print("Stacked test data: ", stacked_test)

    meta_alg = LinearRegression()

    return meta_alg.fit(stacked_valid, y_valid), stacked_test


train_data, test_data = prepare_data()

ridge_model, ridge_metrics = select_pipeline(train_data, ridge)

xgb_model, xgb_metrics = select_pipeline(train_data, xgb_regressor)

dt_model, dt_metrics = select_pipeline(train_data, decision_tree_regressor)

linear_model, lr_metrics = select_pipeline(train_data, linear)

lasso_model, lasso_metrics = select_pipeline(train_data, lasso)

labels = ['xgb', 'decision-tree', 'ridge', 'linear', "lasso"]
mins = [np.min(xgb_metrics),
        np.min(dt_metrics),
        np.min(ridge_metrics),
        np.min(lr_metrics),
        np.min(lasso_metrics)]

index = np.arange(len(labels))

plt.bar(index, mins)

plt.xlabel('Learners')
plt.ylabel('RMSLE')
plt.xticks(index, labels, fontsize=10)
plt.show()

meta_model, stacked_test = stacking(
    train_data,
    Y,
    test_data,
    [xgb_model, dt_model, ridge_model, linear_model, lasso_model])

predict(meta_model, stacked_test)
