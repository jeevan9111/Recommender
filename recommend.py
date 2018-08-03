import math
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import coo_matrix
from tensorflow.contrib.factorization.python.ops import factorization_ops


TEST_SET_RATIO = 10

user_index_recom = 10
user_rated_items = None
Ratings = None
implicit = 0


def make_data():
    dataFrame = pd.read_csv('u.csv', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'],
                            dtype={
                                'user_id': np.int32,
                                'item_id': np.int32,
                                'rating': np.float32,
                                'timestamp': np.int32,
                            })
    dataFrame = dataFrame.values
    userIDs = dataFrame[:, 0]
    itemIDs = dataFrame[:, 1]
    ratedValues = dataFrame[:, 2]

    ratings = np.zeros((userIDs.shape[0], 3), dtype=object)
    unique_users = np.unique(userIDs)
    unique_items = np.unique(itemIDs)

    totalUsers = unique_users.shape[0]
    totalItems = unique_items.shape[0]
    maximumUserID = int(unique_users[-1])
    maximumItemID = int(unique_items[-1])

    if totalUsers != maximumUserID or totalItems != maximumItemID:
        z = np.zeros(maximumUserID + 1, dtype=int)
        z[unique_users] = np.arange(maximumUserID)
        u_r = z[userIDs]

        z = np.zeros(maximumItemID + 1, dtype=int)
        z[unique_items] = np.arange(totalItems)
        i_r = z[itemIDs]

        ratings[:, 0] = u_r
        ratings[:, 1] = i_r
        ratings[:, 2] = ratedValues
    else:
        ratings = dataFrame
        ratings[:, 0] -= 1
        ratings[:, 1] -= 1

    testIndices = sorted(np.random.choice(range(len(ratings)), size=int(len(ratings) / 10), replace=False))
    testData = ratings[testIndices]
    trainData = np.delete(ratings, testIndices, axis=0)

    u_train, i_train, r_train = trainData[:, 0], trainData[:, 1], trainData[:, 2]
    train_sparse = coo_matrix((r_train, (u_train, i_train)), shape=(maximumUserID, totalItems))
    u_test, i_test, r_test = testData[:, 0], testData[:, 1], testData[:, 2]
    test_sparse = coo_matrix((r_test, (u_test, i_test)), shape=(maximumUserID, totalItems))

    return ratings, ratings[:, 0], ratings[:, 1], train_sparse, test_sparse


def train_model(data):
    row_factor, col_factor = None, None
    wt_type = 0
    num_rows, num_cols = data.shape

    row_wts = np.ones(num_rows)
    col_wts = None
    times_rated = np.array((data > 0.0).sum(0)).transpose()

    if wt_type == implicit:
        frac = []
        for i in times_rated:
            if i != 0:
                frac.append(1.0 / i)
            else:
                frac.append(0.0)
        col_wts = np.array(np.power(frac, 0.08)).flatten()
    else:
        col_wts = np.array(100 * times_rated).flatten()

    with tf.Graph().as_default():
        input_tensor = tf.SparseTensor(indices=list(zip(data.row, data.col)), values=(data.data).astype(np.float32),
                                       dense_shape=data.shape)

        model = factorization_ops.WALSModel(num_rows, num_cols, n_components=10, unobserved_weight=0.001,
                                            regularization=0.08, row_weights=row_wts, col_weights=col_wts)
        row_factor = model.row_factors[0]
        col_factor = model.col_factors[0]

    sess = tf.Session(graph=input_tensor.graph)

    with input_tensor.graph.as_default():
        row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
        col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

        sess.run(model.initialize_op)
        sess.run(model.worker_init)
        for _ in range(20):
            sess.run(model.row_update_prep_gramian_op)
            sess.run(model.initialize_row_update_op)
            sess.run(row_update_op)
            sess.run(model.col_update_prep_gramian_op)
            sess.run(model.initialize_col_update_op)
            sess.run(col_update_op)
    output_row = row_factor.eval(session=sess)
    output_col = col_factor.eval(session=sess)
    sess.close()
    return output_row, output_col


def save_model(ratings, user_map, item_map, row_factor, col_factor):
    model_dir = 'model'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)
    np.save(os.path.join(model_dir, 'ratings'), ratings)
    np.save(os.path.join(model_dir, 'user'), user_map)
    np.save(os.path.join(model_dir, 'item'), item_map)
    np.save(os.path.join(model_dir, 'row'), row_factor)
    np.save(os.path.join(model_dir, 'col'), col_factor)


def generate_recommendations(user_idx, row_factor, col_factor, k):
    user_rated = [i[1] for i in Ratings if i[0] == user_idx]

    assert (row_factor.shape[0] - len(user_rated)) >= k
    user_f = row_factor[user_idx]
    pred_ratings = col_factor.dot(user_f)
    k_r = k + len(user_rated)
    candidate_items = np.argsort(pred_ratings)[-k_r:]
    recommended_items = [i for i in candidate_items if i not in user_rated]
    recommended_items = recommended_items[-k:]
    recommended_items.reverse()
    return recommended_items


def get_rmse(output_row, output_col, actual):
    mse = 0
    for i in range(actual.data.shape[0]):
        row_pred = output_row[actual.row[i]]
        col_pred = output_col[actual.col[i]]
        err = actual.data[i] - np.dot(row_pred, col_pred)
        mse += err * err
    mse /= actual.data.shape[0]
    rmse = math.sqrt(mse)
    return rmse


Ratings, user_map, item_map, tr_sparse, test_sparse = make_data()
output_row, output_col = train_model(tr_sparse)
save_model(Ratings, user_map, item_map, output_row, output_col)
train_rmse = get_rmse(output_row, output_col, tr_sparse)
test_rmse = get_rmse(output_row, output_col, test_sparse)

print(generate_recommendations(10, output_row, output_col, 5))
