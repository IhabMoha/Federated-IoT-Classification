# Federated-IoT-Classification

This repo for preprocessing and IoT device classification in Machine Learning (ML) and Federated Learning (FL). The dataset used in this repo is available [here](https://iotanalytics.unsw.edu.au/iottraces.html). This repo is structured as follows:
1. **Preprocessing:** a number of bash scripts that use Cisco Systems Joy tool for collecting flow information from the IoT dataset.
2. **Feature Extraction:** configurable Python code for extracting features from the generated flows in the preprocessing part.
3. **Machine Learning (ML):** Python code that uses TensorFlow and Keras for training a model for IoT device classification.
4. **Federated Learning (FL):** Python code that uses TnesorFlow Federated library for IoT device classification in federated settings.
