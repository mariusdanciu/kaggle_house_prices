import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from scipy.stats import pearsonr
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import math
import warnings
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, input_cols=[]):
        self.input_cols = input_cols
        self.classes_ = []
        self.in_cols_range = range(len(self.input_cols))

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.classes_ = [np.unique(X[:, i]) for i in self.input_cols]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        xcp = X.copy()
        classes = [np.unique(X[:, i]) for i in self.input_cols]
        for i in self.in_cols_range:
            if len(np.intersect1d(classes[i], self.classes_[i])) < len(classes[i]):
                diff = np.setdiff1d(classes[i], self.classes_[i])
                raise ValueError("X[%d] contains new labels: %s" % (i, str(diff)))
            else:
                xcp[:, self.input_cols[i]] = np.searchsorted(self.classes_[i], X[:, self.input_cols[i]])
        return pd.DataFrame(xcp)


warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
pd.set_option('display.max_rows', 1000)

trainingFile = "./data/train.csv"
testFile = "./data/test.csv"

trainDf = pd.read_csv(trainingFile, header = 0)
testDf = pd.read_csv(testFile, header = 0)

trainDf['MoSold'] = trainDf['MoSold'].apply(str)
testDf['MoSold'] = testDf['MoSold'].apply(str)


target = 'SalePrice'

Y = trainDf[target]

training = trainDf.drop(['Id', target], axis=1)
testing = testDf.drop(['Id'], axis=1)

cat_features = []
cat_features_idx = []
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

#enc = ce.OrdinalEncoder(cols = str_cols)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
norm = Normalizer('l2')
xgb_reg = xgboost.XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
                               learning_rate=0.08, max_delta_step=0, max_depth=5,
                               min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
                               objective='reg:linear', reg_alpha=0, reg_lambda=1,
                               scale_pos_weight=1, seed=0, silent=True, subsample=0.75)

lr = LinearRegression()


# display  -------------------------------------------------------------------------------------------------------------


print(training.dtypes)
print(training.head(10))


enc = MultiLabelEncoder(input_cols = np.array(str_cols_idx))

# correlations  --------------------------------------------------------------------------------------------------------

t_pipe = Pipeline(steps = [
    ('catencode', enc),
    ('null_handler', imp)])


fit_pipeline = t_pipe.fit(pd.concat([training, testing], axis=0))

transformed = fit_pipeline.transform(training)

t_df = pd.DataFrame(data = transformed, columns = training.columns)

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


# train with lr  -------------------------------------------------------------------------------------------------------

rmses = []
pipelines = []
range = [0, 0.01, 0.05, 1.0, 1.5, 2.0]

# transform_pipeline = Pipeline(steps = [
#     ('catencode', enc),
#     ('null_handler', imp)])
#
# fe = transform_pipeline.fit(training, Y)

for l in range:
    print("Train with alpha ", l)
    ridge_reg = Ridge(alpha=l, copy_X=True, fit_intercept=True, max_iter=None,
                      normalize=True, random_state=None, solver='auto', tol=0.001)

    # Play more here
    xgb_reg.reg_lambda = l
    xgb_reg.reg_alpha = l

    pipeline = Pipeline(steps = [
        #('fe', fe),
        ('lr', xgb_reg)])

    kf = KFold(n_splits = 10, random_state = 0)

    iter_rmse = []
    iteration = 0
    for train_idx, test_idx in kf.split(t_df):
        print("KFold iteration ", iteration )
        X_train, X_test = t_df.iloc[train_idx], t_df.iloc[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        model = pipeline.fit(X_train, y_train)
        y_predict = model.predict(X_test)

        mse = mean_squared_log_error(y_test, y_predict)
        rmse = math.sqrt(mse)
        print(rmse)
        iter_rmse.append(rmse)
        iteration += 1

    rmses.append(np.mean(iter_rmse))
    pipelines.append(pipeline)

min_index = np.argmin(rmses)
print('Min RMSE index: ', min_index)


best_pipeline = pipelines[min_index]
print('Best pipeline', best_pipeline)

print(rmses)
plt.plot(range, rmses)
plt.show()

# predict --------------------------------------------------------------------------------------------------------------

sp_id = testDf['Id']

m = best_pipeline.fit(t_df, Y)

testing_transformed = pd.DataFrame(data = t_pipe.transform(testing), columns = testing.columns)

pred = m.predict(testing_transformed)


result = pd.DataFrame({'Id': sp_id, 'SalePrice': pred}, index = None)

print(result.head(10))
result.to_csv('./submission.csv', index = False)


