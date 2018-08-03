import numpy as np
from flask import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/<int:user_idx>')
def get(user_idx):
    Ratings = np.load('model/ratings.npy')
    row_factor = np.load('model/row.npy')
    col_factor = np.load('model/col.npy')
    user_rated = [i[1] for i in Ratings if i[0] == user_idx]

    assert (row_factor.shape[0] - len(user_rated)) >= 5
    user_f = row_factor[user_idx]
    pred_ratings = col_factor.dot(user_f)
    k_r = 5 + len(user_rated)
    candidate_items = np.argsort(pred_ratings)[-k_r:]
    recommended_items = [i for i in candidate_items if i not in user_rated]
    recommended_items = recommended_items[-5:]
    recommended_items.reverse()
    print(recommended_items)
    recommendations = [{str(i): int(item)} for i, item in enumerate(recommended_items)]
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
