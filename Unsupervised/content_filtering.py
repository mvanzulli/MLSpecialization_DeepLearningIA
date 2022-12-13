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
'Backward and forwawrd pass'
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)

'Test the outputoff'

model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)