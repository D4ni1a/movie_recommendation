# Movie recommendation system

Shulepin Danila (BS21-DS02)

d.shulepin@innopolis.university

## Task description

A recommender system is a type of information filtering system that suggests items or content to users based on their interests, preferences, or past behavior. These systems are commonly used in various domains, such as e-commerce, entertainment, social media, and online content platforms.

Your assignment is to create a recommender system of movies for users.

## Data description

### Main raw dataset

The dataset is [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) consisting user ratings to movies.

* It consists of 100,000 ratings from 943 users on 1682 movies
* Ratings are ranged from 1 to 5
* Each user has rated at least 20 movies
* It contains simple demographic info for the users (age, gender, occupation, zip code)

### Detailed data description

Here are brief descriptions of the data.

ml-data.tar.gz   -- Compressed tar file.  To rebuild the u data files do this:
                gunzip ml-data.tar.gz
                tar xvf ml-data.tar
                mku.sh

u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
	         user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC   

u.info     -- The number of users, items, and ratings in the u data set.

u.item     -- Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.

u.genre    -- A list of the genres.

u.user     -- Demographic information about the users; this is a tab
              separated list of
              user id | age | gender | occupation | zip code
              The user ids are the ones used in the u.data data set.

u.occupation -- A list of the occupations.

u1.base    -- The data sets u1.base and u1.test through u5.base and u5.test
u1.test       are 80%/20% splits of the u data into training and test data.
u2.base       Each of u1, ..., u5 have disjoint test sets; this if for
u2.test       5 fold cross validation (where you repeat your experiment
u3.base       with each training and test set and average the results).
u3.test       These data sets can be generated from u.data by mku.sh.
u4.base
u4.test
u5.base
u5.test

ua.base    -- The data sets ua.base, ua.test, ub.base, and ub.test
ua.test       split the u data into a training set and a test set with
ub.base       exactly 10 ratings per user in the test set.  The sets
ub.test       ua.test and ub.test are disjoint.  These data sets can
              be generated from u.data by mku.sh.

allbut.pl  -- The script that generates training and test sets where
              all but n of a users ratings are in the training data.

mku.sh     -- A shell script to generate all the u data sets from u.data.

## Structure of repository

```
movie_recommendation
├── README.md               # The README
│
├── data
│   ├── interim             # Intermediate data that has been transformed
│   └── raw                 # The original data
│
├── models                  # Trained and serialized models, final checkpoints
│
├── notebooks               #  Jupyter notebooks      
│
├── references              # Reference list
│
├── reports
│   ├── figures             # Generated graphics and figures to be used in reporting
│   └── final_report.pdf    # Report containing solution description
│
└── benchmark
    ├── data                # data used for evaluation
    ├── evaluate.py         # script that performs evaluation of the given model
    └── get.sh              # download raw data for evaluation
```

## Requirements

Requirements may be found in requirements.txt

Install by:

> pip install requirements.txt

## How to run?

In order to run the project, clone repository localy and install requirements.

In you same directory as movie_recommendation run:

> sh movie_recommendation/benchmark/get.sh
> 
> python3 movie_recommendation/benchmark/evaluate.py

Note that in evaluate.py recomendation given for user with id 1

# Reference:
[1] - F. M. Harper and J. A. Konstan, “The MovieLens datasets: History and context,” ACM Trans. Interact. Intell. Syst., vol. 5, no. 4, pp. 1–19, 2016. http://dx.doi.org/10.1145/2827872

[2] - F. M. Harper and J. A. Konstan, “The MovieLens datasets: History and context,” ACM Trans. Interact. Intell. Syst., vol. 5, no. 4, pp. 1–19, 2016.

[3] - R. Vidiyala, “How to build a movie recommendation system,” Towards Data Science, 02-Oct-2020. [Online]. Available: https://towardsdatascience.com/how-to-build-a-movie-recommendation-system-67e321339109. [Accessed: 03-Dec-2023].

[4] - P. Aher, “Evaluation metrics for recommendation systems — an overview,” Towards Data Science, 09-Aug-2023. [Online]. Available: https://towardsdatascience.com/evaluation-metrics-for-recommendation-systems-an-overview-71290690ecba. [Accessed: 03-Dec-2023].

[5] - “Recommendation system in python,” GeeksforGeeks, 18-Jul-2021. [Online]. Available: https://www.geeksforgeeks.org/recommendation-system-in-python/. [Accessed: 03-Dec-2023].
