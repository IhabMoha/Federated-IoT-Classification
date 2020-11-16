# Budgeted Online Selection of Candidate Clients to Participate in Federated Learning [paper](https://ieeexplore.ieee.org/document/9249424)

This repo provides code dealing with IoT dataset for:
1. **Preprocessing using Cisco Systems Joy tool**
2. **Features Extraction using configurable Python code**
3. **Classification using Deep Machine Learning (TensorFlow and Keras)**
4. **Classification using Federated Learning (Random and Online algorithms implemented using TnesorFlow Federated Library)**

The dataset used in this repo is available [here](https://iotanalytics.unsw.edu.au/iottraces.html).
To start, do the following steps:
1. Create pcap_files, json_files, and csv_files folders.
2- Download all pcap files from the dataset website to the pcap_files folder (2016->20 files, 2018->27 files).
3- Run joy_v2_IoT_devices_2016.sh (in shell-scripts folder) to generate json files in the json_files folder for the 2016 dataset
4- Run joy_v2_IoT_devices_2018.sh (in shell_scripts folder) to generate json files in the json_files folder for the 2018 dataset
*Note: regarding steps 3 and 4, joy will generate one json file for every iot device per pcap file*
