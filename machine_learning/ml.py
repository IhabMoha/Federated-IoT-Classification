# Divide dataset into a number of smaller datasets (clients) equal in size for both train and test

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def print_line():
    print('------------------------------------------------------------')

print('Using TensorFlow version: ', tf.__version__)
print('Using Pandas version: ', pd.__version__)

# Load data from file using Pandas
IoT_features = pd.read_csv('/home/user/csv_files/features_2016_10M.csv', sep=',')

############################################################
# Normalization
############################################################
print_line()
print('Normalization ...')
# Isolate labels from features
IoT_features_labels = IoT_features.pop('device_co')
ary = IoT_features.values
min_max_scaler = preprocessing.MinMaxScaler()
scaled_ary = min_max_scaler.fit_transform(ary)
#IoT_features = pd.DataFrame(scaled_ary, columns = ['total_sleep_time', 'total_active_time', 'total_flow_volume',
# 'flow_rate', 'avg_packet_size', 'num_servers', 'num_protocols', 'uniq_dns', 'dns_interval', 'ntp_interval'])
IoT_features = pd.DataFrame(scaled_ary, columns = ['total_sleep_time', 'total_active_time', 'total_flow_volume',
 'flow_rate', 'avg_packet_size', 'num_servers', 'num_protocols', 'uniq_dns', 'dns_interval', 'ntp_interval'])
IoT_features['device_co'] = IoT_features_labels
#print(IoT_features)


############################################################
# Split data into n_clients Train and Test
############################################################
n_clients = 1
# Split data to Train and Test
IoT_features_size = len(IoT_features)
print_line()
print('Dataset size: ', IoT_features_size)

# Mix the whole array to reaarange different devices, not just inside the loop per device!
train_IoT_features = [[] for x in range(0, n_clients)]
test_IoT_features = [[] for x in range(0, n_clients)]
print('Splitting dataset into training and test ...')
for co in range(1, 25):
  co_data = IoT_features.query('device_co == ' + str(co))
  # Reindex
  co_data.reindex(np.random.permutation(co_data.index))
  #print(co_data)
  co_data_size = len(co_data.index)
  #print('Data size', co_data_size)
  co_data_train_size = int(0.70 * co_data_size)
  #print('Train size:', co_data_train_size)
  co_data_test_size = int(0.30 * co_data_size)
  #print('Test size:', co_data_test_size)

  # Split the train dataset into n_clients datasets
  co_data_train_size_per_client = int(co_data_train_size / n_clients)
  #print('Size of train dataset for device', co, 'is', co_data_train_size_per_client)
  co_train_IoT_features = []
  co_start = 0
  co_end = co_data_train_size_per_client
  #print(co_start, co_end)
  for client_co in range(0, n_clients):
    co_train_IoT_features.append(co_data[co_start:co_end])
    co_start = co_end
    co_end = co_end + co_data_train_size_per_client
  
  # Split the test dataset into n_clients datasets
  co_data_test_size_per_client = int(co_data_test_size / n_clients)
  #print('Size of test dataset for device', co, 'is', co_data_test_size_per_client)
  co_test_IoT_features = []
  # Use co_start from last time
  co_end = co_start + co_data_test_size_per_client
  #print(co_start, co_end)
  for client_co in range(0, n_clients):
    co_test_IoT_features.append(co_data[co_start:co_end])
    co_start = co_end
    co_end = co_end + co_data_test_size_per_client
  
  if co == 1:
    for client_co in range(0, n_clients):
      train_IoT_features[client_co] = co_train_IoT_features[client_co]
      test_IoT_features[client_co] = co_test_IoT_features[client_co]

  else:
    for client_co in range(0, n_clients):
      train_IoT_features[client_co] = train_IoT_features[client_co].append(co_train_IoT_features[client_co])
      test_IoT_features[client_co] = test_IoT_features[client_co].append(co_test_IoT_features[client_co])

# Reindexing datasets to start from 0 and go sequentially
print_line()
print('Reindexing ...')
for client_co in range(0, n_clients):
  train_IoT_features[client_co].reset_index(drop=True, inplace=True)
  test_IoT_features[client_co].reset_index(drop=True, inplace=True)

print('Train size', len(train_IoT_features[0]))

############################################################
# Split labels from Train and Test datasets
############################################################
# Split labels
print_line()
print('Splitting labels ...')
train_IoT_features_labels = [[] for x in range(0, n_clients)]
test_IoT_features_labels = [[] for x in range(0, n_clients)]
for client_co in range(0, n_clients):
  train_IoT_features_labels[client_co] = train_IoT_features[client_co].pop('device_co')
  test_IoT_features_labels[client_co] = test_IoT_features[client_co].pop('device_co')


############################################################
# Convert Train and Test datasets into tf.data.Dataset
############################################################
# Convert pandas dataframes to tf.data.Database

for c_co in range(0, n_clients):
  train_dataset = tf.data.Dataset.from_tensor_slices((train_IoT_features[c_co].to_numpy(), train_IoT_features_labels[c_co].to_numpy()))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_IoT_features[c_co].to_numpy(), test_IoT_features_labels[c_co].to_numpy()))

  nu_of_epoches = 35
  batch_size = 3
  ready_train_dataset = train_dataset.repeat(nu_of_epoches).shuffle(len(train_IoT_features[c_co])).batch(batch_size)
  ready_test_dataset = test_dataset.batch(batch_size)

  print_line()
  print('Creating mode ...')
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(25, activation = tf.nn.relu),
      tf.keras.layers.Dense(25, activation = tf.nn.relu),
      tf.keras.layers.Dense(25, activation = tf.nn.softmax)
  ])

  print_line()
  print('Compiling model ...')
  model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

  print_line()
  print('Training model ...')
  model.fit(ready_train_dataset, epochs = nu_of_epoches, steps_per_epoch = int(len(train_IoT_features[c_co])/batch_size))
  print_line()
  print('Evaluating model ...')
  test_loss, test_accuracy = model.evaluate(ready_test_dataset)
  print_line()
  print('Accuracy on test dataset:', test_accuracy)
