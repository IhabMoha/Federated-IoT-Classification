'''
+ Multiple runs to generate confidence intervals
+ Auto produce results for multiple selected clients and multiple r2 values.
+ Proposed algorithm: find better than best (found between 1 and \alpha) without updating the threshold
+ Best (offline) algrithm: Have to test all clients, sort them based on accuracy, then select the top ones.
+ Random algorithm create a fixed list just like the secretary but select nodes randomly with probability < Sel/N
+ Proposed algorithm works as follows:
1-  Test the first \alpha clients and find the best client with highest accuracy
    NOTE: clients (thin and fat) are distributed randomly in the list of clients
2-  For the rest of clients, find best clients that have accuracy higher than the one you find in (1).
    NOTE: if you can't find those client, then take them from the end of the list
3-  Start the proposed with the list of those best clients in (2) and run for 20/30/.. Epochs
4-  Start random as usual for 20/30/... Epochs
5- Compare.

Test is 20% of data and divided equally on multiple Federated datasets
80% Train
For Thin (X1% of data and X2% of devices per client) clients:
    (1) Data size per client = X1% of Train
    (2) Number of devices per client = X2% of devices
    (3) Data per device = (1) / (2)
    For each client:
        (4) Get random sample of (2) devices:
        For each device in (4):
            Get (3) randomly selected from Train
For Fat (Y1% of data and Y2% of devices per client) clients:
    Revise the abobe for Thin to be used here for Fat
'''
import pandas as pd
from sklearn import preprocessing
import numpy as np, scipy.stats as st
import matplotlib.pyplot as plt
import math
from statistics import mean
import sys
import random

import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff
import os
import time
import csv

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("output.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

sys.stdout = Logger()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(13)
random.seed(33)
if six.PY3: 
    print('Python3')
    tff.framework.set_default_executor(tff.framework.create_local_executor())

def print_line():
    print('------------------------------------------------------------')

print_line()
print('TensorFlow Version: ', tf.__version__)
print_line()
print('TensorFlow Federated Version: ', tff.__version__)
print_line()
print(tff.federated_computation(lambda: 'TensorFlow Federated is working -_-')())

# Load data from file using Pandas
IoT_features = pd.read_csv('/home/user/csv_files/features_2016_10Min.csv', sep=',')
print_line()


############################################################
# Normalization
############################################################
print('Normalization ...')
# Isolate labels from features
IoT_features_labels = IoT_features.pop('device_co')
# Normalize
ary = IoT_features.values
min_max_scaler = preprocessing.MinMaxScaler()
scaled_ary = min_max_scaler.fit_transform(ary)
IoT_features = pd.DataFrame(scaled_ary, columns = ['total_sleep_time', 'total_active_time', 'total_flow_volume',
 'flow_rate', 'avg_packet_size', 'num_servers', 'num_protocols', 'uniq_dns', 'dns_interval', 'ntp_interval'])
# Attach labels
IoT_features['device_co'] = IoT_features_labels
print_line()


############################################################
# Parameters
############################################################
print('Parameters ...')
print_line()

IoT_features_size = len(IoT_features)
org_nu_devices = 24
n_devices = org_nu_devices-5 # 5 devices will be removed duing the device loop due to lack of suficient data

n_clients = 800

no_comm_rounds = 20
client_no_of_epoches = 8
batch_size = 3

no_test_datasets = 1
test_ratio = 0.20
train_ratio = 0.80
train_data_size = int(train_ratio * IoT_features_size)
test_data_size = int(test_ratio * IoT_features_size)

thin_clients_ratio = 0.80
fat_clients_ratio = 0.20
no_thin_clients = int(n_clients * thin_clients_ratio)
no_fat_clients = n_clients - no_thin_clients

thin_train_data_ratio = 0.01
fat_train_data_ratio = 0.10
datasize_per_thin = int(thin_train_data_ratio * train_data_size)
datasize_per_fat = int(fat_train_data_ratio * train_data_size)

thin_device_ratio = 0.20
fat_device_ratio = 0.80
no_devices_per_thin = int(n_devices * thin_device_ratio)
if no_devices_per_thin == 0:
    no_devices_per_thin = 1
no_devices_per_fat = int(n_devices * fat_device_ratio)
if no_devices_per_fat == 0:
    no_devices_per_fat = 1

datasize_per_device_thin = int(datasize_per_thin / no_devices_per_thin)
datasize_per_device_fat = int(datasize_per_fat / no_devices_per_fat)

print('Number of devices:', n_devices)
print('Number of clients:', n_clients)
print('Number of communication rounds:', no_comm_rounds)
print('Number of epochs per client:', client_no_of_epoches)
print('Bacth size:', batch_size)

print('Dataset size: ', IoT_features_size)
print('Number of test datasets:', no_test_datasets)
print('Ratio of Train dataset:', train_ratio, 'with', train_data_size, 'records')
print('Ratio of Test dataset:', test_ratio, 'with', test_data_size, 'records')

print('There are', no_thin_clients, 'Thin clients, a ratio of', thin_clients_ratio,'% of clients')
print('There are', no_fat_clients, 'Fat clients, a ratio of', fat_clients_ratio,'% of clients')

print('A Thin client can have', thin_train_data_ratio, '% of train data with', datasize_per_thin, 'records')
print('A Fat client can have', fat_train_data_ratio, '% of train data with', datasize_per_fat, 'records')

print('A Thin client can have data from', no_devices_per_thin, 'devices (', thin_device_ratio, '%)')
print('A Fat client can have data from', no_devices_per_fat, 'devices (', fat_device_ratio, '%)')

print('A Thin client can have', datasize_per_device_thin, 'records per device')
print('A Fat client can have', datasize_per_device_fat, 'records per device')

test_IoT_features = [[] for x in range(0, no_test_datasets)]
train_IoT_per_device = [[] for x in range(0, n_devices)]
train_IoT_features = [[] for x in range(0, n_clients)]
print_line()


############################################################
# Get data per device and isolate 20% for Test dataset
############################################################
print('Getting data per device and creating Test dataset ...')
# Start per device processing
device_co = 0
for co in range(1, (org_nu_devices+1)):
    co_data = IoT_features.query('device_co == ' + str(co))
    #print(co_data)
    co_data_size = len(co_data)
    #print('Device:', co, 'Data size', co_data_size)
    if co_data_size < 100:
        #print('Skipping this device due to lack of enough data')
        continue
    # Reindex
    co_data.reindex(np.random.permutation(co_data.index))
    co_data_train_size = int(train_ratio * co_data_size)
    #print('Train size:', co_data_train_size)
    co_data_test_size = int(test_ratio * co_data_size)
    #print('Test size:', co_data_test_size)

    # -------------------- TEST --------------------
    co_data_test_size_per_dataset = int(co_data_test_size/no_test_datasets)
    co_start = 0
    co_end = co_data_test_size_per_dataset
    for test_co in range(0, no_test_datasets):
        if len(test_IoT_features[test_co]) == 0:
            test_IoT_features[test_co] = co_data[co_start:co_end]
        else:
            test_IoT_features[test_co] = test_IoT_features[test_co].append(co_data[co_start:co_end])
        co_start = co_end
        co_end = co_end + co_data_test_size_per_dataset

    # -------------------- TRAIN --------------------
    # Use co_start with last value
    train_IoT_per_device[device_co] = co_data[co_start:]
    #print('Device', device_co,'has', len(train_IoT_per_device[device_co]), 'training records')

    device_co += 1
print_line()


############################################################
# Split Train data over Thin and Fat clients
############################################################
print('Splitting Train dataset over Thin and Fat clients ...')
fat_dic = {}
thin_dic = {}
for client_co in range(0, n_clients):
    if client_co < no_thin_clients:
        # -------------------- Thin clients
        rand_devices = random.sample(range(0, n_devices), no_devices_per_thin)
        for device_id in rand_devices:
            # Stats
            if device_id not in thin_dic:
                thin_dic[device_id] = 1
            else:
                thin_dic[device_id] += 1
            # -------------------
            dev_data_size = len(train_IoT_per_device[device_id])
            if dev_data_size-datasize_per_device_thin < 0:
                print('Error: data_start for device', device_id, 'is negative')
                print('Device data size:', dev_data_size, 'per device data size for thin clients', datasize_per_device_thin)
                sys.exit(1)
            data_start = random.randint(0, (dev_data_size-datasize_per_device_thin))
            if len(train_IoT_features[client_co]) == 0:
                train_IoT_features[client_co] = train_IoT_per_device[device_id][data_start:(data_start+datasize_per_device_thin)]
            else:
                train_IoT_features[client_co] = train_IoT_features[client_co].append(train_IoT_per_device[device_id][data_start:(data_start+datasize_per_device_thin)])
    else:
        # -------------------- Fat clients
        rand_devices = random.sample(range(0, n_devices), no_devices_per_fat)      
        for device_id in rand_devices:
            # Stats
            if device_id not in fat_dic:
                fat_dic[device_id] = 1
            else:
                fat_dic[device_id] += 1
            # -------------------
            dev_data_size = len(train_IoT_per_device[device_id])
            if dev_data_size-datasize_per_device_fat < 0:
                print('Error: data_start for device', device_id, 'is negative')
                print('Device data size:', dev_data_size, 'per device data size for fat clients', datasize_per_device_fat)
                sys.exit(1)
            data_start = random.randint(0, (dev_data_size-datasize_per_device_fat))
            if len(train_IoT_features[client_co]) == 0:
                train_IoT_features[client_co] = train_IoT_per_device[device_id][data_start:(data_start+datasize_per_device_fat)]
            else:
                train_IoT_features[client_co] = train_IoT_features[client_co].append(train_IoT_per_device[device_id][data_start:(data_start+datasize_per_device_fat)])
print_line()
print('Thin dictionary:')
for key in sorted(thin_dic):
    print((key, thin_dic[key]), end="")
print('\nFat dictionary:')
for key in sorted(fat_dic):
    print((key, fat_dic[key]), end="")
print()
print_line()


############################################################
# Reindexing
############################################################
# Drop old index and use an index that starts from 0 and go sequentially
# Because index comes from different devices, so index is: 1,2,3,4,100,101,102,103,201,...
# Will be 0,1,2,3,4,5,6,7,8,9,...
print('Reindexing ...')
for client_co in range(0, n_clients):
    train_IoT_features[client_co].reset_index(drop=True, inplace=True)
for test_co in range(0, no_test_datasets):
    test_IoT_features[test_co].reset_index(drop=True, inplace=True)

print_line()
'''
for co in range(0, n_clients):
    print('Size of client', co ,'dataset:', len(train_IoT_features[co]))
for test_co in range(0, no_test_datasets):
    print('Size of test', test_co ,'dataset:', len(test_IoT_features[test_co]))
print_line()
'''


############################################################
# Split labels from Train and Test datasets
############################################################
# Split labels
print('Splitting labels ...')
train_IoT_features_labels = [[] for x in range(0, n_clients)]
test_IoT_features_labels = [[] for x in range(0, no_test_datasets)]
for client_co in range(0, n_clients):
    train_IoT_features_labels[client_co] = train_IoT_features[client_co].pop('device_co')
for test_co in range(0, no_test_datasets):
    test_IoT_features_labels[test_co] = test_IoT_features[test_co].pop('device_co')
print_line()


############################################################
# Convert Train and Test datasets into tf.data.Dataset
############################################################
print('Converting train and test DataFrames to tf.data.Dataset ...')
clients_train_dataset = [[] for x in range(0, n_clients)]
clients_test_dataset = [[] for x in range(0, no_test_datasets)]
for client_co in range(0, n_clients):
    clients_train_dataset[client_co] = tf.data.Dataset.from_tensor_slices((train_IoT_features[client_co].to_numpy(), train_IoT_features_labels[client_co].to_numpy()))
    clients_train_dataset[client_co] = clients_train_dataset[client_co].repeat(client_no_of_epoches).shuffle(len(train_IoT_features[client_co])).batch(batch_size)
for test_co in range(0, no_test_datasets):
    clients_test_dataset[test_co] = tf.data.Dataset.from_tensor_slices((test_IoT_features[test_co].to_numpy(), test_IoT_features_labels[test_co].to_numpy()))
    clients_test_dataset[test_co] = clients_test_dataset[test_co].batch(batch_size)
print_line()


############################################################
# Federated Averaging Functions
############################################################
def create_compiled_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(25, activation = tf.nn.relu),
        tf.keras.layers.Dense(25, activation = tf.nn.relu),
        tf.keras.layers.Dense(25, activation = tf.nn.softmax)
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),#reduction=tf.keras.losses.Reduction.NONE),
        #optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        optimizer='adam',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        #metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])
    return model

def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_train_batch)


############################################################
# Federated Evaluation Functions
############################################################
# Create a copy of the learning model without optimization to be used in evaluation
def create_compiled_keras_model_for_evaluation():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(25, activation = tf.nn.relu),
        tf.keras.layers.Dense(25, activation = tf.nn.relu),
        tf.keras.layers.Dense(25, activation = tf.nn.softmax)
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),#reduction=tf.keras.losses.Reduction.NONE),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        #metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])
    return model

def model_fn_evaluation():
    keras_model = create_compiled_keras_model_for_evaluation()
    return tff.learning.from_compiled_keras_model(keras_model, sample_test_batch)


############################################################
# Build Federated Averaging
############################################################
# Sample training batch
sample_train_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(clients_train_dataset[n_clients-1]).next())
sample_test_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(clients_test_dataset[0]).next())
print_line()
print('Build fedearted averaging process ...')
#iterative_process = tff.learning.build_federated_averaging_process(model_fn)
print_line()


############################################################
# Federated Iterative Process
############################################################
# Create random list of clients indices
print('Randomize list of clients:')
client_random_list = random.sample(range(0, n_clients), n_clients)
print(client_random_list)


#------------------------------------------------
# Start testing clients and record accuracies
#------------------------------------------------
clients_acc_file_name = '/home/shadha/Dropbox/Development/IoT/csv_files/Accuracy_N_' + str(n_clients)
clients_acc_file_name += '_Rounds_' + str(no_comm_rounds) +'_Epochs_' + str(client_no_of_epoches)
clients_acc_file_name += '_Clients_Fat' + str(fat_clients_ratio) +'_Thin_' + str(thin_clients_ratio)
clients_acc_file_name += '_Data_Fat_' + str(fat_train_data_ratio) +'_Thin_' + str(thin_train_data_ratio) + '.csv'

# ---- Either test all clients and save accuracy per client in a file or load clients' accuracy from a file
create_load = 'load'

if create_load == 'create':
    # Training each client
    print('Testing clients and recording accuracy...')
    start_time = time.time()
    clients_acc_csv = open(clients_acc_file_name, 'w')
    clients_acc_writer = csv.writer(clients_acc_csv, delimiter=',')
    clients_acc_writer.writerow(['Client_ID', 'Accuracy'])
    clients_test_acc_list = []
    for client_co in range(0, n_clients):
        # Initialize iterative process
        iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        state = iterative_process.initialize()
        client_id = client_random_list[client_co]
        print('Client ID:', client_id)
        client_dataset = [clients_train_dataset[client_id]]
        state, metrics = iterative_process.next(state, client_dataset)
        print('Metrics=', metrics)
        # Create the evaluation model
        evaluation = tff.learning.build_federated_evaluation(model_fn_evaluation)
        # Evaluate on test dataset
        test_metrics = evaluation(state.model, clients_test_dataset)
        print('Test evaluation:', test_metrics)
        clients_test_acc_list.append([client_id, test_metrics[0]])
        clients_acc_writer.writerow([client_id, test_metrics[0]])
    end_time = time.time()
    print('Testing clients execution time (minutes):', (end_time-start_time)/60)
else:
    # ---- Or load the list from a file if already produced
    print('Loading clients\' accuracy ...')
    clients_acc_csv = open(clients_acc_file_name, 'r')
    clients_acc_reader = csv.reader(clients_acc_csv, delimiter=',')
    next(clients_acc_reader)
    clients_test_acc_list = []
    for row in clients_acc_reader:
        clients_test_acc_list.append([int(row[0]), float(row[1])])
clients_acc_csv.close()

# Sorting the clients accuracy list in a new list (original list is required in the proposed)
sorted_clients_acc_list = sorted(clients_test_acc_list ,key = lambda x:x[1], reverse=True)

active_clients_list = [10, 20, 30, 40, 50]
no_Rs = len(active_clients_list)
all_results_list = []

# Run the experiment 5 times, save the results every time and compute the confidence interval
no_runs = 5
for run_co in range(1, no_runs+1):
    print('Run number', run_co)
    print_line()
    # Fix seeds every run
    np.random.seed(13)
    random.seed(33)
    results_list = []
    result_co = 0

    start_time = time.time()
    for no_of_active_client in active_clients_list:
        print('Number of selected (active) clients:', no_of_active_client)
        print_line()

        #------------------------------------------------
        # Start the optimal algorithm
        #------------------------------------------------
        print('Starting the Optimal algorithm ...')
        optimal_clients = []
        optimal_clients_acc = []
        optimal_train_dataset = []
        fat_co = 0
        for co in range(no_of_active_client):
            optimal_clients.append(sorted_clients_acc_list[co][0])
            optimal_clients_acc.append(sorted_clients_acc_list[co][1])
            optimal_train_dataset.append(clients_train_dataset[optimal_clients[co]])
            if optimal_clients[co] >= no_thin_clients:
                fat_co += 1
        thin_co = no_of_active_client - fat_co
        print('Optimal selected clients:', optimal_clients)
        print('Optimal accuracies      :', optimal_clients_acc)
        print('Average accuracy:', mean(optimal_clients_acc))
        print('Training ...')
        iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        state = iterative_process.initialize()
        for round_num in range(1, no_comm_rounds+1):
            state, metrics = iterative_process.next(state, optimal_train_dataset)
            print('round {:2d}, metrics={}'.format(round_num, metrics))
        # Create the evaluation model
        evaluation = tff.learning.build_federated_evaluation(model_fn_evaluation)
        # On test
        test_metrics = evaluation(state.model, clients_test_dataset)
        print('Test evaluation:', test_metrics)
        results_list.append(['opt', no_of_active_client, mean(optimal_clients_acc), test_metrics[0], fat_co, thin_co])
        print_line()


        #------------------------------------------------
        # Start the random algorithm
        #------------------------------------------------
        print('Starting the random algorithm ...')
        # Create the random list
        rand_clients = []
        rand_clients_acc = []
        sel_co = 0
        fat_co = 0
        prop_val = no_of_active_client/n_clients
        print('Propability of selection:', prop_val)
        for co in range(n_clients):
            #print(co, ':')
            gen_prop = random.random()
            #print('Propability:', gen_prop)
            if gen_prop < prop_val:
                rand_clients.append(client_random_list[co])
                rand_clients_acc.append(clients_test_acc_list[co][1])
                if client_random_list[co] >= no_thin_clients:
                    fat_co += 1
                #print('Selecting this client')
                sel_co += 1
                if sel_co == no_of_active_client:
                    break
            elif (n_clients-co) == (no_of_active_client-len(rand_clients)):
                rand_clients.append(client_random_list[co])
                rand_clients_acc.append(clients_test_acc_list[co][1])
                if client_random_list[co] >= no_thin_clients:
                    fat_co += 1
                #print('Adding from the end of list')
                sel_co += 1
                if sel_co == no_of_active_client:
                    break
        thin_co = no_of_active_client - fat_co
        print_line()
        print('Random selected clients:', rand_clients)
        print('Random accuracies      :', rand_clients_acc)
        print('Average accuracy:', mean(rand_clients_acc))
        print_line()
        random_train_dataset = []
        for co in range(no_of_active_client):
            random_train_dataset.append(clients_train_dataset[rand_clients[co]])
        iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        state = iterative_process.initialize()
        for round_num in range(1, no_comm_rounds+1):
            state, metrics = iterative_process.next(state, random_train_dataset)
            print('round {:2d}, metrics={}'.format(round_num, metrics))
        # Create the evaluation model
        evaluation = tff.learning.build_federated_evaluation(model_fn_evaluation)
        # On test
        test_metrics = evaluation(state.model, clients_test_dataset)
        print('Test evaluation:', test_metrics)
        results_list.append(['rnd', no_of_active_client, mean(rand_clients_acc), test_metrics[0], fat_co, thin_co])
        print_line()


        #------------------------------------------------
        # Start the proposed algorithm
        #------------------------------------------------
        print('Starting the Proposed algorithm ...')

        alpha_list = []
        r2 = 1
        alpha_list.append(int(n_clients * math.exp(-math.pow((math.factorial(r2)), (1/r2)))))
        r2 = 2
        alpha_list.append(int(n_clients * math.exp(-math.pow((math.factorial(r2)), (1/r2)))))
        r2 = 3
        alpha_list.append(int(n_clients * math.exp(-math.pow((math.factorial(r2)), (1/r2)))))
        r2 = 4
        alpha_list.append(int(n_clients * math.exp(-math.pow((math.factorial(r2)), (1/r2)))))
        r2 = 5
        alpha_list.append(int(n_clients * math.exp(-math.pow((math.factorial(r2)), (1/r2)))))

        for alpha in alpha_list:
            print('Alpha:', alpha)    
            print('Step 1: Serach for the client with highest acuuracy (', alpha, ' clients) ...')
            #----------------------------------------------
            # Step 1: Serach for the client with highest accuracy from the first N/e clients
            best_client_id = -1
            best_client_acc = 0
            for client_co in range(0, alpha):
                client_id = client_random_list[client_co]
                #print('Client ID:', client_id, clients_test_acc_list[client_co][0])
                acc = clients_test_acc_list[client_co][1]
                #print('Test evaluation:', acc)
                # Check for best and worst accuracy and clients
                if acc > best_client_acc:
                    best_client_acc = acc
                    best_client_id = client_id
                #print_line()

            print('Best client:', best_client_id, 'with test accuracy: ', best_client_acc)
            print_line()

            #----------------------------------------------
            # Step 2: Find the best clients
            prop_clients = []
            prop_clients_acc = []
            best_client_co = 0
            random_client_co = 0
            fat_co = 0
            print('Finding the best clients ...')
            for client_co in range(alpha, n_clients):
                client_id = client_random_list[client_co]
                #print('Client ID:', client_id, clients_test_acc_list[client_co][0])
                acc = clients_test_acc_list[client_co][1]
                #print('Test evaluation:', acc)
                
                if acc > best_client_acc:
                    if len(prop_clients) < no_of_active_client:
                        prop_clients.append(client_id)
                        prop_clients_acc.append(acc)
                        best_client_co += 1
                        if client_id >= no_thin_clients:
                            fat_co += 1
                        #print('Adding this best client')
                        if len(prop_clients) >= no_of_active_client:
                            break
                elif (n_clients - client_co) <= (no_of_active_client - len(prop_clients)):
                    prop_clients.append(client_id)
                    prop_clients_acc.append(acc)
                    random_client_co += 1
                    if client_id >= no_thin_clients:
                        fat_co += 1
                    #print('Adding this client due to lack of best clients')
                #print_line()
            thin_co = no_of_active_client - fat_co
            print(best_client_co, 'Best clients found')
            print(random_client_co, 'Random clients found from the end of the list')
            print_line()
            print('Proposed selected clients:', prop_clients)
            print('Proposed accuracies      :', prop_clients_acc)
            print('Average accuracy:', mean(prop_clients_acc))
            print_line()

            #----------------------------------------------
            # Step 3: Start the proposed algorithm
            print('Training ...')
            # Initialize iterative process
            iterative_process = tff.learning.build_federated_averaging_process(model_fn)
            state = iterative_process.initialize()
            clients_datasets = []
            for co in range(no_of_active_client):
                client_id = prop_clients[co]
                clients_datasets.append(clients_train_dataset[client_id])

            for round_num in range(1, no_comm_rounds+1):
                state, metrics = iterative_process.next(state, clients_datasets)
                print('round {:2d}, metrics={}'.format(round_num, metrics))
            # Create the evaluation model
            evaluation = tff.learning.build_federated_evaluation(model_fn_evaluation)
            # On test
            test_metrics = evaluation(state.model, clients_test_dataset)
            print('Test evaluation:', test_metrics)
            results_list.append(['prp', no_of_active_client, alpha, mean(prop_clients_acc), test_metrics[0], fat_co, thin_co])
            print_line()

        print('Optimal algorithm Avg accuracy:', results_list[result_co][2], 'Accuracy:', results_list[result_co][3], 'Fats:', results_list[result_co][4], 'Thins:', results_list[result_co][5])
        print('Proposed algorithm (r2=1) alpha:', results_list[result_co+2][2], 'Avg accuracy: :', results_list[result_co+2][3], 'Accuracy:', results_list[result_co+2][4], 'Fats:', results_list[result_co+2][5], 'Thins:', results_list[result_co+2][6])
        print('Proposed algorithm (r2=2) alpha:', results_list[result_co+3][2], 'Avg accuracy: :', results_list[result_co+3][3], 'Accuracy:', results_list[result_co+3][4], 'Fats:', results_list[result_co+3][5], 'Thins:', results_list[result_co+3][6])
        print('Proposed algorithm (r2=3) alpha:', results_list[result_co+4][2], 'Avg accuracy: :', results_list[result_co+4][3], 'Accuracy:', results_list[result_co+4][4], 'Fats:', results_list[result_co+4][5], 'Thins:', results_list[result_co+4][6])
        print('Proposed algorithm (r2=4) alpha:', results_list[result_co+5][2], 'Avg accuracy: :', results_list[result_co+5][3], 'Accuracy:', results_list[result_co+5][4], 'Fats:', results_list[result_co+5][5], 'Thins:', results_list[result_co+5][6])
        print('Proposed algorithm (r2=5) alpha:', results_list[result_co+6][2], 'Avg accuracy: :', results_list[result_co+6][3], 'Accuracy:', results_list[result_co+6][4], 'Fats:', results_list[result_co+6][5], 'Thins:', results_list[result_co+6][6])
        print('Random algorithm Avg accuracy:', results_list[result_co+1][2], 'Accuracy:', results_list[result_co+1][3], 'Fats:', results_list[result_co+1][4], 'Thins:', results_list[result_co+1][5])
        result_co += 7

    all_results_list.append(results_list)
    # Save results
    results_file_name = '/home/shadha/Dropbox/Development/IoT/Results/Results_Run_' + str(run_co)
    results_file_name += '_N_' + str(n_clients)
    results_file_name += '_Rounds_' + str(no_comm_rounds) +'_Epochs_' + str(client_no_of_epoches)
    results_file_name += '_Clients_Fat' + str(fat_clients_ratio) +'_Thin_' + str(thin_clients_ratio)
    results_file_name += '_Data_Fat_' + str(fat_train_data_ratio) +'_Thin_' + str(thin_train_data_ratio) + '.csv'

    results_csv = open(results_file_name, 'w')
    results_writer = csv.writer(results_csv, delimiter=',')

    # 1- Save Alpha
    results_writer.writerow(['', '', '', '', 'Alpha Values', '', '', '', ''])
    results_writer.writerow(['Prp_r2=1', 'Prp_r2=2', 'Prp_r2=3', 'Prp_r2=4', 'Prp_r2=5'])
    results_writer.writerow([results_list[2][2], results_list[3][2], results_list[4][2], results_list[5][2], results_list[6][2]])
    results_writer.writerow([''])

    # 2- Save accuracies
    results_writer.writerow(['', '', '', '', 'Accuracies', '', '', '', ''])
    results_writer.writerow(['N', 'R', 'Opt', 'Rnd', 'Prp_r2=1', 'Prp_r2=2', 'Prp_r2=3', 'Prp_r2=4', 'Prp_r2=5'])
    result_co = 0
    for co in range(no_Rs):
        results_writer.writerow([n_clients, results_list[result_co][1], \
                                results_list[result_co][3], results_list[result_co+1][3], \
                                results_list[result_co+2][4], results_list[result_co+3][4], results_list[result_co+4][4], \
                                results_list[result_co+5][4], results_list[result_co+6][4]])
        result_co += 7

    # 3- Save Avg accuracies
    results_writer.writerow([''])
    results_writer.writerow(['', '', '', '', 'Avg Accuracies', '', '', '', ''])
    results_writer.writerow(['N', 'R', 'Opt', 'Rnd', 'Prp_r2=1', 'Prp_r2=2', 'Prp_r2=3', 'Prp_r2=4', 'Prp_r2=5'])
    result_co = 0
    for co in range(no_Rs):
        results_writer.writerow([n_clients, results_list[result_co][1], \
                                results_list[result_co][2], results_list[result_co+1][2], \
                                results_list[result_co+2][3], results_list[result_co+3][3], results_list[result_co+4][3], \
                                results_list[result_co+5][3], results_list[result_co+6][3]])
        result_co += 7

    # 4- Save Fat nodes
    results_writer.writerow([''])
    results_writer.writerow(['', '', '', '', '#Fat Nodes', '', '', '', ''])
    results_writer.writerow(['N', 'R', 'Opt', 'Rnd', 'Prp_r2=1', 'Prp_r2=2', 'Prp_r2=3', 'Prp_r2=4', 'Prp_r2=5'])
    result_co = 0
    for co in range(no_Rs):
        results_writer.writerow([n_clients, results_list[result_co][1], \
                                results_list[result_co][4], results_list[result_co+1][4], \
                                results_list[result_co+2][5], results_list[result_co+3][5], results_list[result_co+4][5], \
                                results_list[result_co+5][5], results_list[result_co+6][5]])
        result_co += 7

    # 4- Save Thin nodes
    results_writer.writerow([''])
    results_writer.writerow(['', '', '', '', '#Thin Nodes', '', '', '', ''])
    results_writer.writerow(['N', 'R', 'Opt', 'Rnd', 'Prp_r2=1', 'Prp_r2=2', 'Prp_r2=3', 'Prp_r2=4', 'Prp_r2=5'])
    result_co = 0
    for co in range(no_Rs):
        results_writer.writerow([n_clients, results_list[result_co][1], \
                                results_list[result_co][5], results_list[result_co+1][5], \
                                results_list[result_co+2][6], results_list[result_co+3][6], results_list[result_co+4][6], \
                                results_list[result_co+5][6], results_list[result_co+6][6]])
        result_co += 7

    results_csv.close()
    print_line()
    # Print results on screen
    print(results_list)
    print_line()
    end_time = time.time()
    print('Execution time in minutes:', (end_time-start_time)/60)
    print_line()
    print_line()

# Build confidence intervals
confid_list = []
result_co = 0
for R_co in range(no_Rs):
    acc_lists = [[] for x in range(7)]
    for run_co in range(no_runs):
        acc_lists[0].append(all_results_list[run_co][result_co][3])
        acc_lists[1].append(all_results_list[run_co][result_co+1][3])
        acc_lists[2].append(all_results_list[run_co][result_co+2][4])
        acc_lists[3].append(all_results_list[run_co][result_co+3][4])
        acc_lists[4].append(all_results_list[run_co][result_co+4][4])
        acc_lists[5].append(all_results_list[run_co][result_co+5][4])
        acc_lists[6].append(all_results_list[run_co][result_co+6][4])
    for co in range(7):
        cf_res = st.t.interval(0.95, len(acc_lists[co])-1, loc=np.mean(acc_lists[co]), scale=st.sem(acc_lists[co]))
        confid_list.append(cf_res)
    result_co += 7

# Save results
results_file_name = '/home/shadha/Dropbox/Development/IoT/Results/Confidence_N_' + str(n_clients)
results_file_name += '_Rounds_' + str(no_comm_rounds) +'_Epochs_' + str(client_no_of_epoches)
results_file_name += '_Clients_Fat' + str(fat_clients_ratio) +'_Thin_' + str(thin_clients_ratio)
results_file_name += '_Data_Fat_' + str(fat_train_data_ratio) +'_Thin_' + str(thin_train_data_ratio) + '.csv'

results_csv = open(results_file_name, 'w')
results_writer = csv.writer(results_csv, delimiter=',')

results_writer.writerow(['', '', '', '', '', '', '', 'Confidence', '', '', '', '', '', '', '', ''])
results_writer.writerow(['N', 'R', 'Opt Lo', 'Opt Up', 'Rnd Lo', 'Rnd Up', 'Prp_r2=1 Lo', 'Prp_r2=1 Up', 'Prp_r2=2 Lo', 'Prp_r2=2 Up', 'Prp_r2=3 Lo', 'Prp_r2=3 Up', 'Prp_r2=4 Lo', 'Prp_r2=4 Up', 'Prp_r2=5 Lo', 'Prp_r2=5 Up'])
result_co = 0
for co in range(no_Rs):
    results_writer.writerow([n_clients, active_clients_list[co], \
                            confid_list[result_co][0], confid_list[result_co][1], confid_list[result_co+1][0], confid_list[result_co+1][1],\
                            confid_list[result_co+2][0], confid_list[result_co+2][1], confid_list[result_co+3][0], confid_list[result_co+3][1],\
                            confid_list[result_co+4][0], confid_list[result_co+4][1], confid_list[result_co+5][0], confid_list[result_co+5][1],\
                            confid_list[result_co+6][0], confid_list[result_co+6][1]])
    result_co += 7
results_csv.close()
