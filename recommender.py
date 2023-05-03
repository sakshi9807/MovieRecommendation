import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

movielens_data = fetch_movielens(min_rating=4.0)

print(repr(movielens_data['train']))
print(repr(movielens_data['test']))

model = LightFM(loss='warp')

model.fit(movielens_data['train'], epochs=30, num_threads=2)

def recommend_movies(model, data, user_ids):
    # number of users and movies in training data
    num_users, num_movies = data['train'].shape
    
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(num_movies))
        top_movies = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")
        for movie in known_positives[:3]:
            print("        %s" % movie)

        print("     Recommended:")
        for movie in top_movies[:3]:
            print("        %s" % movie)

recommend_movies(model, movielens_data, [3, 25, 451])