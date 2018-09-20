import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
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

#trainDf['OverallQual'] = trainDf['OverallQual'].pow(3)
# #trainDf['GrLivArea'] = trainDf['GrLivArea'].pow(3) #nope
# trainDf['GarageCars'] = trainDf['GarageCars'].pow(2)
#trainDf['GarageArea'] = trainDf['GarageArea'].pow(2)
#
#testDf['OverallQual'] = testDf['OverallQual'].pow(3)
# #testDf['GrLivArea'] = testDf['GrLivArea'].pow(3)
# testDf['GarageCars'] = testDf['GarageCars'].pow(2)
#testDf['GarageArea'] = testDf['GarageArea'].pow(2)

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

    return (fit_pipeline, pd.DataFrame(data=transformed, columns=training.columns),
            pd.DataFrame(data=testing, columns=testing.columns))


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
    kf = KFold(n_splits=5, random_state=0)
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


def train(t_df, regularizations, make_pipeline):
    rmsles = []
    pipelines = []

    for l in regularizations:
        print("Training with lambda: ", l)
        # Play more here

        pipeline = make_pipeline(l)

        mean = cv(t_df, pipeline)

        print("Mean RMSLE: ", mean)
        rmsles.append(mean)
        pipelines.append(pipeline)

    min_index = np.argmin(rmsles)
    print('Min RMSLE: ', np.min(rmsles))
    print('Min RMSLE index: ', min_index)

    best_pipeline = pipelines[min_index]
    print('Best pipeline', best_pipeline)

    best_model = best_pipeline.fit(t_df, Y)

    print("RMSLES : ", rmsles)

    return (best_model, rmsles)


def decition_tree_regressor(l):
    dtr = DecisionTreeRegressor(criterion='mse', max_depth=7, max_features=None,
                                max_leaf_nodes=None, min_impurity_decrease=0.0,
                                min_impurity_split=None, min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                presort=False, random_state=0, splitter='best')
    return Pipeline(steps=[
        ('lr', dtr)])


def xgb_regressor(l):
    xgb_reg = xgboost.XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.7, gamma=0,
                                   learning_rate=0.05, max_delta_step=0, max_depth=5,
                                   min_child_weight=4, missing=None, n_estimators=500, nthread=-1,
                                   objective='reg:linear', reg_alpha=0, reg_lambda=l,
                                   scale_pos_weight=1, seed=0, silent=True, subsample=0.75)

    return Pipeline(steps=[
        ('lr', xgb_reg)])


def ridge(l):
    reg = Ridge(alpha=l, copy_X=False, fit_intercept=True, max_iter=None,
                normalize=True, random_state=0, solver='auto', tol=0.001)
    return Pipeline(steps=[
        ('lr', reg)])


def linear(l):
    reg = LinearRegression(normalize=True)
    return Pipeline(steps=[
        ('lr', reg)])


def predict(model, testing, t_pipe):
    sp_id = testDf['Id']

    testing_transformed = pd.DataFrame(data=t_pipe.transform(testing), columns=testing.columns)

    pred = model.predict(testing_transformed)

    result = pd.DataFrame({'Id': sp_id, 'SalePrice': pred}, index=None)

    print(result.head(10))
    result.to_csv('./submission.csv', index=False)
    print("Submission file created")


t_pipe, train_data, test_data = prepare_data()

# correlations(train_data, )

regs = [0, 0.5, 0.7, 1.0, 1.5, 2]

print("HAS NaN: ", train_data.isnull().sum())

xgb_model, rmsles1 = train(train_data, regs, xgb_regressor)

dt_model, rmsles2 = train(train_data, regs, decition_tree_regressor)

ridge_model, rmsles3 = train(train_data, regs, ridge)

linear_model, rmsles4 = train(train_data, regs, linear)

plt.plot(regs, rmsles1, 'k-o',
         regs, rmsles2, 'r-o',
         regs, rmsles3, 'g-o',
         regs, rmsles4, 'y-o'
         )
plt.legend(['xgb', 'decision-tree', 'ridge', 'linear'], loc='best')
plt.xlabel('Lambda regularization')
plt.ylabel('RMSLE')
plt.show()

# predict(xgb_model, test_data, t_pipe)
