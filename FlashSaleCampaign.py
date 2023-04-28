# Packages imports
"""
Ryan Mokarian, 2023-01-20
"""

from numpy.random import seed
seed(1)
import numpy as np
import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


if __name__ == "__main__":

    df_all = pd.read_csv('data_train.csv')#.drop('Unnamed: 0', axis=1).sort_index()

    print(len(list(set(df_all['bfgudid'])))) # 8000
    print(df_all.head())
    print('df Shape :', df_all.shape) # (255696, 13)
    print('NaN values :', df_all.isna().sum()) # all 0, product, currency, gross_usd_amount, price_tier = 176329
    print('Number of unique users: ', len(set(df_all['bfgudid']))) # 8000
    sns.countplot(x='bfgudid', data=df_all)
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('Users Sessions Histogram')
    plt.title('Users Sessions Histogram')
    # plt.show()
    plt.close()
    df_all_count = df_all['bfgudid'].value_counts()

    # remove all users without a complete price_tier data on the sessions
    cond = df_all['price_tier'].isna()
    df_nanRemoved = df_all[~cond]
    df_reduced_feat = df_nanRemoved.drop(columns = ['app_store','country_code','product','currency', 'gross_usd_amount'])
    print(df_reduced_feat.shape) # (79367, 9)
    df_count = df_reduced_feat['bfgudid'].value_counts()
    print('Users with min number of plays: ',  df_count.min()) # 1
    print('Median number of plays: ', df_count.median()) # 5
    print('Max number of plays: ', df_count.max()) # 259

    # remove users with less than 5 entries (median)
    users_more_than_4_sessions = list(df_count.loc[df_count>=5].index)
    df_var_ses = df_reduced_feat.loc[df_reduced_feat['bfgudid'].isin(users_more_than_4_sessions)]
    print(df_var_ses.shape) # (69210, 9)


    # only keep users's first 5 entries and store them in training and testing dataframes where
    users_id_list = list(set(df_var_ses['bfgudid']))
    print(len(users_id_list)) # 4296

    df_train = pd.DataFrame(columns=list(df_var_ses.columns))
    df_test = pd.DataFrame(columns=list(df_var_ses.columns))
    df_notTrain_notTest = df_test = pd.DataFrame(columns=list(df_var_ses.columns))
    training_end_date = pd.to_datetime('2021-01-20', format='%Y-%m-%d')
    for id in users_id_list:
        cond = (df_var_ses['bfgudid'] == id)
        dfi = df_var_ses[cond].sort_values('activity_date').reset_index(drop=True)[:5]
        dfi['activity_date'] = pd.to_datetime(dfi['activity_date'], format='%Y-%m-%d')
        if dfi.loc[4,'activity_date'] <= training_end_date:
            df_train = pd.concat([df_train, dfi])
        elif dfi.loc[0,'activity_date'] > training_end_date:
            df_test = pd.concat([df_test, dfi])
        else: df_notTrain_notTest = pd.concat([df_notTrain_notTest, dfi])

    df_train = df_train.reset_index(drop=True) # for Jan 24: 17690/5 = 3538; for Jan 20: 14830/5 = 2966
    df_test = df_test.reset_index(drop=True) # for Jan 24: 745/5 = 149; for Jan 20: 2780/5 = 556; 556/(556+2966) = 16%
    df_notTrain_notTest = df_notTrain_notTest.reset_index(drop=True) # for Jan 24: 3045/5 = 609; for Jan 20: 3870/5 = 774
    #
    # __________________________________________________________________________________________________________________
    # This section is to save the time of testing by commenting from teh beginning to the line that creates csv files
    df_train.to_csv('df_train_5entries.csv')
    df_test.to_csv('df_test_5entries.csv')
    df_train = pd.read_csv('df_train_5entries.csv')
    df_test = pd.read_csv('df_test_5entries.csv')
    # __________________________________________________________________________________________________________________

    # Frame the date for supervised learning. Note delete the fifth row as it connects the last entry of each user to the first one of the next user
    df_train = df_train.drop(columns = ['Unnamed: 0','activity_date'], axis=1).set_index('bfgudid', drop=True)
    df_train = df_train[['price_tier'] + [col for col in df_train.columns if col != 'price_tier']]

    df_test = df_test.drop(columns = ['Unnamed: 0','activity_date'], axis=1).set_index('bfgudid', drop=True)
    df_test = df_test[['price_tier'] + [col for col in df_test.columns if col != 'price_tier']]

    # Reframe the training data for lstm (scale and add next value of the dependent variable as the last column)
    df2 = df_train.copy()
    values = df2.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[7, 8, 9, 10, 11]], axis=1, inplace=True)
    # drop rows that are multipliers of 5 as they include dependent variable belong to two users
    train = reframed[reframed.index % 5 != 0]
    print(train.head())

    # Reframe the testing data for lstm (scale and add next value of the dependent variable as the last column)
    df3 = df_test.copy()
    values2 = df3.values
    # ensure all data is float
    values2 = values2.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled2 = scaler.fit_transform(values2)
    # frame as supervised learning
    reframed2 = series_to_supervised(scaled2, 1, 1)
    # drop columns we don't want to predict
    reframed2.drop(reframed2.columns[[7, 8, 9, 10, 11]], axis=1, inplace=True)
    # drop rows that are multipliers of 5 as they include dependent variable belong to two users
    test = reframed2[reframed2.index % 5 != 0]
    print(test.head())

    # split into input and outputs
    train_X, train_y = train.values[:, :-1], train.values[:, -1]
    test_X, test_y = test.values[:, :-1], test.values[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) # (11864, 1, 6) (11864,) (2224, 1, 6) (2224,)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('LSTM_loss')
    plt.title('Loss (MAE)')
    # plt.show()
    plt.close()

    # Evaluate Model
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    print("R2 score:", r2_score(inv_y, inv_yhat))
