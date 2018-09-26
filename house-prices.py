import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression, LassoCV
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor

from MLE import MultiLabelEncoder

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
pd.set_option('display.max_rows', 1000)

trainingFile = "./data/train.csv"
testFile = "./data/test.csv"

trainDf = pd.read_csv(trainingFile, header=0)
testDf = pd.read_csv(testFile, header=0)

trainDf['MoSold'] = trainDf['MoSold'].apply(str)
testDf['MoSold'] = testDf['MoSold'].apply(str)

target = 'SalePrice'

Y = trainDf[target]


def prepare_data():
    training = trainDf.drop(['Id'], axis=1)
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

    # treat NaN as a different category
    for c in str_cols:
        training[c] = training[c].fillna("$NULL")
        testing[c] = testing[c].fillna("$NULL")

    # training = training.drop(training[(training['GrLivArea']>4000) & (training['SalePrice']<300000)].index)
    training = training.drop([target], axis=1)

    print(training.dtypes)
    print(training.head(10))

    enc = MultiLabelEncoder(input_cols=np.array(str_cols_idx))

    t_pipe = Pipeline(steps=[
        ('catencode', enc),
        ('null_handler', Imputer(missing_values='NaN', strategy='mean', axis=0))
    ])

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
    kf = KFold(n_splits=10, random_state=10)
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
    for split in range(2, 10):
        est = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                                    max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, min_samples_leaf=1,
                                    min_samples_split=split, min_weight_fraction_leaf=0.0,
                                    presort=False, random_state=10, splitter='best')
        pipelines.append(Pipeline(steps=[('DecisionTreeRegressor', est)]))
    return pipelines


def xgb_regressor():
    pipelines = []
    for l in [0, 0.5, 0.7, 1.0, 1.5, 2]:
        for d in range(3, 10):
            est = xgboost.XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8, gamma=0,
                                       learning_rate=0.1, max_delta_step=0, max_depth=d,
                                       min_child_weight=10, missing=None, n_estimators=300, nthread=-1,
                                       objective='reg:linear', reg_alpha=0.2, reg_lambda=l,
                                       scale_pos_weight=1, seed=0, silent=True, subsample=0.8)

        pipelines.append(Pipeline(steps=[('XGBRegressor', est)]))

    return pipelines


def ridge():
    pipelines = []
    for l in [0, 0.5, 0.7, 1.0, 1.5, 2]:
        est = Ridge(alpha=l, copy_X=True, fit_intercept=True, max_iter=None,
                    normalize=True, random_state=10, solver='auto', tol=0.0001)
        pipelines.append(Pipeline(steps=[('Ridge', est)]))
    return pipelines


def linear():
    pipelines = []
    est = LinearRegression(normalize=True)
    pipelines.append(Pipeline(steps=[('LinearRegression', est)]))
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
    kf = KFold(n_splits=5, random_state=10)

    validation_body = {
    }
    test_body = {
    }
    for p in pipelines:
        validation_body['pred_' + p.steps[0][0]] = np.zeros(len(training.index))
        test_body['pred_' + p.steps[0][0]] = np.zeros(len(test.index))

    valid_df = pd.DataFrame(validation_body)

    test_df = pd.DataFrame(test_body)

    for train_idx, validation_idx in kf.split(training):
        x_train, x_validation = training.iloc[train_idx], training.iloc[validation_idx]
        y_train, y_validation = y[train_idx], y[validation_idx]

        for col, alg in enumerate(pipelines):
            t_cpy = x_train.copy()
            alg_model = alg.fit(t_cpy, y_train)

            valid_pred = alg_model.predict(x_validation)

            valid_df.iloc[validation_idx, col] = valid_pred

    for col, alg in enumerate(pipelines):
        t_cpy = training.copy()
        alg_model = alg.fit(t_cpy, y)
        test_pred = alg_model.predict(test)

        test_df.iloc[:, col] = test_pred

    meta_alg = LinearRegression(normalize=True)

    return meta_alg.fit(valid_df, y), test_df


train_data, test_data = prepare_data()

correlations(train_data)

print("Run Lasso")
lasso_model = Pipeline(steps=[('lassocv', LassoCV(cv=10, random_state=10))]).fit(train_data, Y)

print("Run DT")
dt_model, dt_metrics = select_pipeline(train_data, decision_tree_regressor)

print("Run XGB")
xgb_model, xgb_metrics = select_pipeline(train_data, xgb_regressor)

print("Run Ridge")
ridge_model, ridge_metrics = select_pipeline(train_data, ridge)

print("Run LR")
linear_model, lr_metrics = select_pipeline(train_data, linear)

labels = ['xgb', 'dt', 'ridge', 'linear']
mins = [np.min(xgb_metrics),
        np.min(dt_metrics),
        np.min(ridge_metrics),
        np.min(lr_metrics)
        ]

index = np.arange(len(labels))

plt.bar(index, mins)

plt.xlabel('Learners')
plt.ylabel('RMSLE')
plt.xticks(index, labels, fontsize=10)
plt.show()

meta_model, stack_test = stacking(
    train_data,
    Y,
    test_data,
    [xgb_model, dt_model, ridge_model, linear_model, lasso_model])

predict(meta_model, stack_test)

# predict(xgb_model, test_data)
