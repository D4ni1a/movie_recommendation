import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])
        
install('surprise')

# import useful libraries
import pickle
import numpy as np
import pandas as pd
from IPython.display import display_html
import warnings
from sklearn.model_selection import train_test_split
from surprise import SVD
import numpy as np
import surprise
from surprise import Reader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

base = "/content/ml-100k/"

ratings_data = pd.read_csv(base + 'u.data', sep = '\t', header = None)
ratings_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']

class Preprocessor:
    def __init__(self, items, users):
        """
        Initialize nessesary models and objects

        :param items: dataframe of movies
        :param users: dataframe of users
        """
        self.reader = Reader(rating_scale=(1,5))

        # SVD model for additional features
        self.svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
        self.users = users
        self.items = items
        self.train_sparse_matrix = None
        self.train_averages = dict()
        self.col_names = ['M', 'F'] + ["occupation_" + str(i + 1) for i in range(len(users['occupation'].unique()))]
        self.encoder = None

    def fit_svd(self, df):
        """
        Fitinf SVD model on train data

        :param df: train dataset
        """
        # Building special form dataset and fit SVD
        train_data_mf = Dataset.load_from_df(df[['user_id', 
                                                           'item_id', 
                                                           'rating']], 
                                                 self.reader)
        trainset = train_data_mf.build_full_trainset()
        self.svd.fit(trainset)
        return trainset
    
    def get_average_ratings(self, of_users = True):
        """
        Calculating average rating for movies and users using User-Movie matrix

        :param of_user: boolean parameter controls switching between users and items
        :return: average rating
        """
        # Choose axes for users or movies
        ax = 1 if of_users else 0 # 1 - User axes, 0 - Movie axes
        sum_of_ratings = self.train_sparse_matrix.sum(axis = ax).A1
        # Whether a user rated that movie or not
        is_rated = self.train_sparse_matrix != 0
        no_of_ratings = is_rated.sum(axis=ax).A1

        # Maximum number of users and movies
        u, m = self.train_sparse_matrix.shape

        # Create a dictionary of users and their average ratings
        # Zero is in case of movie not presenting in ratings
        average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i] if no_of_ratings[i] !=0 else 0
                        for i in range(u if of_users else m)}
        return average_ratings

    def top_users_rates(self, user, movie, k = 5):
        """
        Get rating for movie from most similar users

        :param user: user id
        :param movie: movie id
        :param k: number of similar users
        :return: rating of the most similar users
        """
        # Find nearest users for our user
        user_sim = cosine_similarity(self.train_sparse_matrix[user], 
                                     self.train_sparse_matrix).ravel()
        # Sort by similarity and remove user himself
        # And take rating for this movie by the most similar users
        top_sim_users = user_sim.argsort()[::-1][1:]
        top_ratings = self.train_sparse_matrix[top_sim_users, movie].toarray().ravel()
        
        # If number of similar users less than k, fill by average for this movie
        top_sim_users_ratings = list(top_ratings[top_ratings != 0][:k])
        top_sim_users_ratings.extend([self.train_averages['movie'][movie]]*(k - len(top_sim_users_ratings)))
        return top_sim_users_ratings

    def top_movie_rates(self, user, movie, k = 5):
        """
        Get rating from user for most similar movies

        :param user: user id
        :param movie: movie id
        :param k: number of similar movies
        :return: rating of the most similar movies
        """
        # Find nearest movies for our movie
        movie_sim = cosine_similarity(self.train_sparse_matrix[:,movie].T, 
                                      self.train_sparse_matrix.T).ravel()
        top_sim_movies = movie_sim.argsort()[::-1][1:]
        # Sort by similarity and remove movie himself
        # And take rating for movies most similar to current
        top_ratings = self.train_sparse_matrix[user, top_sim_movies].toarray().ravel()
        
        # If number of similar movies less than k, fill by average for this user
        top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:k])
        top_sim_movies_ratings.extend([self.train_averages['user'][user]]*(k - len(top_sim_movies_ratings)))
        return top_sim_movies_ratings

    def one_hot_encoding(self, df, encoder = None):
        """
        OneHot encoding

        :param df: dataframe to be encoded
        :param encoder: onehot encoder itself
        :return: encoder and encoded dataframe
        """
        # Converting type of columns to category
        df['gender'] = df['gender'].astype('category')
        df['occupation'] = df['occupation'].astype('category')

        if encoder:
            # For existing encoder
            enc_data = pd.DataFrame(encoder.transform(
            df[['gender', 'occupation']]).toarray(), columns=self.col_names)
        else:
            # Create new encoder and fit it
            encoder = OneHotEncoder()
            enc_data = pd.DataFrame(encoder.fit_transform(
            df[['gender', 'occupation']]).toarray(), columns=self.col_names)

        final_df = df.join(enc_data).drop(['gender', 'occupation'], axis = 1)

        return encoder, final_df

    def best_values(self, df, rated = False, k = 5):
        """
        Additional features based on similar user, similar movie and their average value

        :param df: dataframe to be encoded
        :param rated: was the data rated
        :param k: numer of similar movies/users
        :return: encoded data
        """
        data = df
        for i in tqdm(data.index):
            # Extract user and movie
            user = data.loc[i]['user_id']
            movie = data.loc[i]['item_id']

            # Get rating of most similar users and movies
            top_users_list = self.top_users_rates(user, movie, k = k)
            top_movie_list = self.top_movie_rates(user, movie, k = k)
            movies_columns = ["M" + str(i+1) for i in range(k)]
            users_columns = ["U" + str(i+1) for i in range(k)]

            # Average this values
            if rated:
                UAvg = data.loc[data['user_id'] == user, 'rating'].mean()
                MAvg = data.loc[data['item_id'] == movie, 'rating'].mean()
            else:
                UAvg = np.mean(top_users_list)
                MAvg = np.mean(top_movie_list)
                
            # Extend initial data
            columns = tuple(users_columns + movies_columns + ['UAvg', 'MAvg'])
            values = top_users_list + top_movie_list + [UAvg, MAvg]
            data.at[i, columns] = values
        return data

    def preprocess(self, df, set_type = 'Train'):
        """
        Preprocessor

        :param df: initial data
        :param set_type: Train/Test/Predict
        :return: preprocessed data
        """
        if set_type == 'Train':
            # Fit SVD, User-Item matrix and OneHot encoder on test data
            trainset = self.fit_svd(df)
            max_users = self.users['user_id'].max()
            max_items = self.items['item_id'].max()
            self.train_sparse_matrix = csr_matrix((df.rating.values,
             (df.user_id.values, df.item_id.values)),
                                 shape = (max_users + 1, max_items + 1))
            self.train_averages['global'] = self.train_sparse_matrix.sum()/self.train_sparse_matrix.count_nonzero()
            self.train_averages['user'] = self.get_average_ratings(of_users = True)
            self.train_averages['movie'] = self.get_average_ratings(of_users = False)
            
            final_train_data = self.best_values(df, rated = True)
            self.encoder, one_hot_train = self.one_hot_encoding(final_train_data)
            one_hot_train = one_hot_train.drop(['timestamp', 'zip code', 'item_title', 'release_date',
                                       'video_release_date', 'IMDb_URL'],
                                      axis=1)
            # Normalize age and add additional features from SVD predictions
            one_hot_train['age'] = one_hot_train['age']/100
            train_preds = self.svd.test(trainset.build_testset())
            train_pred_mf = np.array([pred.est for pred in train_preds])
            one_hot_train = one_hot_train.join(pd.DataFrame(train_pred_mf, columns=['pred']))
            return one_hot_train

        elif set_type == 'Test':
            # Use OneHot encoder to encode gender and occupation
            final_test = self.best_values(df, rated = True)
            _, one_hot_test = self.one_hot_encoding(final_test, encoder = self.encoder)
            one_hot_test = one_hot_test.drop(['timestamp', 'zip code', 'item_title', 'release_date',
                                       'video_release_date', 'IMDb_URL'],
                                      axis=1)
            # Normalize age and add additional features from SVD predictions
            one_hot_test['age'] = one_hot_test['age']/100

            test_data_mf = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], self.reader)
            testset = test_data_mf.build_full_trainset()
            test_preds = self.svd.test(testset.build_testset())
            test_pred_mf = np.array([pred.est for pred in test_preds])
            one_hot_test = one_hot_test.join(pd.DataFrame(test_pred_mf, columns=['pred']))
            return one_hot_test

        else:
            # Use OneHot encoder to encode gender and occupation
            final_test = self.best_values(df, rated = False)
            _, one_hot_pred = self.one_hot_encoding(final_test, encoder = self.encoder)
            rank = pd.DataFrame([0 for i in range(len(one_hot_pred))], columns=['ranking'])
            one_hot_pred = one_hot_pred.join(pd.DataFrame(rank, columns=['rating']))

            # Normalize age and add additional features from SVD predictions
            test_data_mf = Dataset.load_from_df(one_hot_pred[['user_id', 'item_id', 'rating']], self.reader)
            testset = test_data_mf.build_full_trainset()
            test_preds = self.svd.test(testset.build_testset())
            test_pred_mf = np.array([pred.est for pred in test_preds])
            one_hot_pred = one_hot_pred.join(pd.DataFrame(test_pred_mf, columns=['pred']))
            one_hot_pred['age'] = (one_hot_pred['age']/100).astype('float')

            one_hot_pred = one_hot_pred.drop(['user_id', 'zip code',
                                                        'item_title', 'release_date',
                                                        'video_release_date', 'IMDb_URL',
                                                        'rating'], axis=1)
            return one_hot_pred

def MAPE_func(true, pred):
    """
    Mean absolute percentage error

    :param true: true labels
    :param predL predicted labels
    """
    return np.mean(np.abs((true - pred)/true)) * 100

def RMSE_func(true, pred):
    """
    Root mean square error

    :param true: true labels
    :param predL predicted labels
    """
    return np.sqrt(np.mean([(true[i] - pred[i])**2 for i in range(len(pred))]))

def predict(user_id, rates, model, k = 5):
    """
    Recommend movies for user

    :param user_id: id of user
    :param rates: data with existing ratings
    :param model: trained model
    :param k: number of recommended movies
    :return: list of movies
    """

    # Find movies which user hasn't watched yet
    movies = preprocessor.items
    users = preprocessor.users
    merged = pd.merge(rates, users)
    movies_list = merged[merged['user_id'] == user_id]['item_id']
    user_unseen = movies[~movies['item_id'].isin(movies_list)]

    # Preprocess data
    merged_data = pd.merge(users.loc[user_id].to_frame().T, user_unseen, how='cross')
    best_data = preprocessor.preprocess(merged_data, set_type = 'Predict')

    # Make model prediction
    predictions = model.predict(best_data.drop(['item_id'], axis = 1))

    # Find the most highly rated items
    dict_preds = {x:predictions[i] for i, x in enumerate(best_data['item_id'])}
    best_films = [a for a,b in sorted(dict_preds.items(), key=lambda x:x[1], reverse=True)][:k]
    return list(movies[movies['item_id'].isin(best_films)]['item_title'])

if __name__ == "__main__":
    # Restore preprocessor
    with open('movie_recommendation/models/preprocessor.pkl', 'rb') as inp:
        preprocessor = pickle.load(inp)

    final_test_data = pd.read_csv('movie_recommendation/benchmark/data/evaluation_interm.csv')

    xgb_model = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15, n_estimators=100)
    xgb_model.load_model("movie_recommendation/models/model.json")

    y_test = final_test_data['rating']
    x_test = final_test_data.drop(['user_id', 'item_id','rating'], axis=1)
    
    # Dictionaries for storing train and test results
    test_results = dict()
    # From the trained model, get the predictions
    y_test_pred = xgb_model.predict(x_test)
    # Get the rmse and mape of train data
    rmse = RMSE_func(y_test.values, y_test_pred)
    mape = MAPE_func(y_test.values, y_test_pred)
    # Store the results in train_results dictionary
    test_results = {'RMSE': rmse, 'MAPE' : mape}
    print(f"\nRMSE: {np.round(rmse, 4)}\nMAPE: {np.round(mape, 4)}\n")
    
    user_idx = 1
    test_items_list = predict(user_idx, ratings_data, xgb_model)
    print(f'\nFor user {user_idx} we recommend to watch:')
    for it in test_items_list:
        print(it)
