# This code was written step by step for a better explanation

from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import math
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import sys, time, resource


def score_accuracy(y, y_pred, **kwargs):
    diff = np.abs(y - y_pred)
    correct = len(np.array(filter(lambda x: x <= 3, diff)))
    return correct/len(y)*100

if __name__ == '__main__':

    # total time
    start_time_all = time.time()

    # Load Model
    model = joblib.load('DecisionTreeRegressor.pkl')

    # Load dataset into panda's dataframe
    arguments = sys.argv[1:]
    df = pd.read_csv(arguments[0], header=0)

    # Index = first column
    indexDF = df.set_index('x001')

    # Drop useless columns
    indexDF.drop(['x067', 'x095', 'x094', 'x096'], axis=1, inplace=True)

    n_rows, n_col = indexDF.shape

    # Handling the attributes with missing values in training
    columns_cat = ['x148', 'x155', 'x162', 'x253', 'x287', 'x302']
    columns_cont = ['x002', 'x003', 'x004', 'x005', 'x041', 'x044', 'x045', 'x057', 'x058', 'x098', 'x222', 'x223',
                    'x234', 'x235', 'x237', 'x238', 'x239', 'x242', 'x255', 'x256', 'x257', 'x259', 'x265', 'x266',
                    'x272', 'x275', 'x288', 'x289', 'x290', 'x293', 'x295', 'x297', 'x304']

    cat_imputation_df = indexDF[columns_cat]
    cont_imputation_df = indexDF[columns_cont]

    # Drop columns with nan
    indexDF.drop(columns_cat, axis=1, inplace=True)
    indexDF.drop(columns_cont, axis=1, inplace=True)

    # Save the column order
    order_col = indexDF.columns.tolist()
    for col in columns_cat:
        order_col.append(col)
    for col in columns_cont:
        order_col.append(col)

    # Get columns with nan
    cols = []
    for c in cat_imputation_df:
        if sum(cat_imputation_df[c].isnull()) == len(indexDF):
            cols.append(c)

    for c in cont_imputation_df:
        if sum(cont_imputation_df[c].isnull()) == len(indexDF):
            cols.append(c)
            #cat_imputation_df[c].replace(to_replace='NaN', value=0, inplace=True)
    if len(cols)>0:
        for c in cols:
            if c in columns_cat:
                columns_cat.remove(c)
            if c in columns_cont:
                columns_cont.remove(c)

    # Replace possible Nan - remaining columns
    indexDF.replace(to_replace='NaN', value=0, inplace=True)

    # Impute
    imputer_cont = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer_cat = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

    cat_df = pd.DataFrame(imputer_cat.fit_transform(cat_imputation_df),
                          columns=columns_cat, index=cat_imputation_df.index)
    cont_df = pd.DataFrame(imputer_cont.fit_transform(cont_imputation_df),
                           columns=columns_cont, index=cont_imputation_df.index)

    # Columns with all values = nan = 0
    ics = pd.DataFrame(df[cols], columns=df[cols].columns, index=indexDF.index)
    if len(cols) > 0:
        ics.replace(to_replace='NaN', value=0, inplace=True)

    # Join columns
    xc = cat_df.join(ics).join(cont_df)
    df_i = indexDF.join(cat_df).join(ics).join(cont_df)

    # Order of columns
    df1 = df_i[order_col]

    # Data
    data_x = df1.drop(['y'], axis=1, inplace=False)
    data_y = df1['y']

    # Evaluate on test
    start_time = time.time()
    predictions = model.predict(data_x)
    print 'Time to evaluate: ', round(time.time() - start_time, 8)
    print 'Memory usage: ', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0, 8)
    model_rmse = math.sqrt(mean_squared_error(data_y, predictions))
    print "RMSE: ", model_rmse
    model_acc = score_accuracy(data_y, predictions)
    print "Accuracy -- abs(difference(y,y_predicted))<3 - in percent: ", model_acc

    # Print the predictions to file
    f = open('/data/predictions.txt', 'w')
    index = data_x.index.tolist()
    f.write('Index: Predictions \n')
    for i in range(n_rows):
        f.write('{0}: {1}\n'.format(index[i], predictions[i]))
    f.close()
    print 'Total time: ', round(time.time() - start_time_all, 8)
