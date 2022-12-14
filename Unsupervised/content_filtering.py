## Import libraries 
import numpy as np
import numpy.ma as ma
import pandas as pd
pd.set_option("display.precision", 1)
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import glob 
import os
from recsysNN_utils import * 

# Load a csv from a folder
data_path = "Unsupervised/data/recomender_systems/content_filtering"
csv_files = glob.glob(os.path.join(data_path,"*.csv"))

# Load data
top10_df = pd.read_csv(csv_files[5])
bygenre_df = pd.read_csv(csv_files[0])


# Training Data
# The movie content provided to the network is a combination of the original data and some 'engineered features'. 
# Recall the feature engineering discussion and lab from Course 1, Week 2, lab 4. 
# The original features are the year the movie was released and the movie's genre's presented as a one-hot vector.
#  There are 14 genres. The engineered feature is an average rating derived from the user ratings. 

# The user content is composed of engineered features. A per genre average rating is computed per user.
#  Additionally, a user id, rating count and rating average are available but not included in the training or prediction content.
#  They are carried with the data set because they are useful in interpreting data.

# The training set consists of all the ratings made by the users in the data set. Some ratings are repeated to boost the number of training examples of underrepresented genre's. The training set is split into two arrays with the same number of entries, a user array and a movie/item array.  

# Below, let's load and display some of the data.


item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items
print(f"Number of training vectors: {len(item_train)}")


pprint_train(user_train, user_features, uvs,  u_s, maxcount=5)

# Content-based filtering with a neural network
# In the collaborative filtering lab, you generated two vectors, a user vector and an item/movie
#  vector whose dot product would predict a rating. The vectors were derived solely from the ratings.   

# Content-based filtering also generates a user and movie feature vector but recognizes there may be other
#  information available about the user and/or movie that may improve the prediction. 
# The additional information is provided to a neural network which then generates the user and movie vector as shown below.


## Scaling data with min max criteria 
# scale training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled    = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))
#ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))

print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))

# Split the data

item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)
print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test data shape: {item_test.shape}")

## 4 - Neural Network for content-based filtering
# Now, let's construct a neural network as described in the figure above.
#  It will have two networks that are combined by a dot product. You will construct the two networks.
#  In this example, they will be identical. Note that these networks do not need to be the same.
#  If the user content was substantially larger than the movie content, you might elect to increase the complexity 
# of the user network relative to the movie network. 
# In this case, the content is similar, so the networks are the same.


num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    #
    tf.keras.layers.Dense(256, activation = 'relu', name = "L1"),
    #
    tf.keras.layers.Dense(128, activation = 'relu', name = "L2"),
    # 
    tf.keras.layers.Dense(num_outputs, activation = 'linear', name = "L3")
])

item_NN = tf.keras.models.Sequential([
    #
    tf.keras.layers.Dense(256, activation = 'relu', name = "L1"),
    #
    tf.keras.layers.Dense(128, activation = 'relu', name = "L2"),
    # 
    tf.keras.layers.Dense(num_outputs, activation = 'linear', name = "L3")
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()


## Compile the model 

tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)

tf.random.set_seed(1)

#Backward and forwawrd pass
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)

#Test the output

model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)

## 5 - Predictions
# Below, you'll use your model to make predictions in a number of circumstances. 

### 5.1 - Predictions for a new user
# First, we'll create a new user and have the model suggest movies for that user.
#  After you have tried this on the example user content, feel free to change the user content to match your own preferences and see what the model suggests. 
# Note that ratings are between 0.5 and 5.0, inclusive, in half-step increments.


new_user_id = 5000
new_rating_ave = 0.0
new_action = 0.0
new_adventure = 5.0
new_animation = 0.0
new_childrens = 0.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 5.0
new_horror = 0.0
new_mystery = 4.0
new_romance = 0.0
new_scifi = 4.0
new_thriller = 4.0
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])


# The new user enjoys movies from the adventure, fantasy genres. Let's find the top-rated movies for the new user.  
# Below, we'll use a set of movie/item vectors, `item_vecs` that have a vector for each movie in the training/test set. 
# This is matched with the new user vector above and the scaled vectors are used to predict ratings for all the movies.


# generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec, len(item_vecs))

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction 
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display

print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10)


### 5.3 - Finding Similar Items
# The neural network above produces two feature vectors, a user feature vector $v_u$, and a movie feature vector, $v_m$. 
# These are 32 entry vectors whose values are difficult to interpret. However, similar items will have similar vectors. 
# This information can be used to make recommendations. 
# For example, if a user has rated "Toy Story 3" highly, one could recommend similar movies by selecting movies with similar
#  movie feature vectors.

# A similarity measure is the squared distance between the two vectors $ \mathbf{v_m^{(k)}}$ and $\mathbf{v_m^{(i)}}$ :
# $$\left\Vert \mathbf{v_m^{(k)}} - \mathbf{v_m^{(i)}}  \right\Vert^2 = \sum_{l=1}^{n}(v_{m_l}^{(k)} - v_{m_l}^{(i)})^2\tag{1}$$


def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """

    return np.sum(np.add(a,-b)**2)


# A matrix of distances between movies can be computed once when the model is trained and then reused for new recommendations without retraining. 
# The first step, once a model is trained, is to obtain the movie feature vector, $v_m$, for each of the movies.
#  To do this, we will use the trained `item_NN` and build a small model to allow us to run the movie vectors through it to generate $v_m$.

input_item_m = tf.keras.layers.Input(shape=(num_item_features))    # input layer
vm_m = item_NN(input_item_m)                                       # use the trained item_NN
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)                        # incorporate normalization as was done in the original model
model_m = tf.keras.Model(input_item_m, vm_m)                                
model_m.summary()


# Once you have a movie model, you can create a set of movie feature vectors by using the model to predict using a set of item/movie vectors as input. 
# `item_vecs` is a set of all of the movie vectors. 
# It must be scaled to use with the trained model. The result of the prediction is a 32 entry feature vector for each movie.


scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")


count = 50  # number of movies to display
dim = len(vms)
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])
        
m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i,0])
    movie2_id = int(item_vecs[min_idx,0])
    disp.append( [movie_dict[movie1_id]['title'], movie_dict[movie1_id]['genres'],
                  movie_dict[movie2_id]['title'], movie_dict[movie1_id]['genres']]
               )
table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
table