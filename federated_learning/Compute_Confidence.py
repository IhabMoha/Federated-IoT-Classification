import numpy as np, scipy.stats as st
import matplotlib.pyplot as plt
import math
from statistics import mean
import sys
import os
import time
import csv

n_clients = 1600
no_comm_rounds = 20
client_no_of_epoches = 8
fat_clients_ratio = 0.20
thin_clients_ratio = 0.80
fat_train_data_ratio = 0.10
thin_train_data_ratio = 0.01

active_clients_list = [10, 20, 30, 40, 50]

no_runs = 5
no_Rs = 5
no_algs = 7
results_list = [[[] for x in range(no_algs)] for y in range(no_Rs)]

# Load results from the 5 files
for run_co in range(no_runs):
    print('Run number', run_co+1)

    # Read results
    results_file_name = '/home/shadha/Dropbox/Development/IoT/Results/Results_Run_' + str(run_co+1)
    results_file_name += '_N_' + str(n_clients)
    results_file_name += '_Rounds_' + str(no_comm_rounds) +'_Epochs_' + str(client_no_of_epoches)
    results_file_name += '_Clients_Fat' + str(fat_clients_ratio) +'_Thin_' + str(thin_clients_ratio)
    results_file_name += '_Data_Fat_' + str(fat_train_data_ratio) +'_Thin_' + str(thin_train_data_ratio) + '.csv'

    results_csv = open(results_file_name, 'r')
    results_reader = csv.reader(results_csv, delimiter=',')
    # Skip
    for co in range(6):
        next(results_reader)
    # Start reading
    r_co = 0
    for row in results_reader:
        for alg_co in range(no_algs):
            results_list[r_co][alg_co].append(float(row[2+alg_co]))
        r_co += 1
        if r_co == no_Rs:
            break
   
    results_csv.close()

for r_co in range(no_Rs):
    print(r_co)
    for alg_co in range(no_algs):
        print(results_list[r_co][alg_co][1], end=' ')
    print()

# Compute mean and confidence intervals
confid_list = [[[] for y in range(no_algs)] for x in range(no_Rs)]
result_co = 0
for r_co in range(no_Rs):
    for alg_co in range(no_algs):
        print(results_list[r_co][alg_co])
        m = np.mean(results_list[r_co][alg_co])
        cf_res = st.t.interval(0.95, len(results_list[r_co][alg_co])-1, loc=m, scale=st.sem(results_list[r_co][alg_co]))
        confid_list[r_co][alg_co].append(m)
        confid_list[r_co][alg_co].append(cf_res[0])
        confid_list[r_co][alg_co].append(cf_res[1])

# Save results
results_file_name = '/home/shadha/Dropbox/Development/IoT/Results/Confidence_N_' + str(n_clients)
results_file_name += '_Rounds_' + str(no_comm_rounds) +'_Epochs_' + str(client_no_of_epoches)
results_file_name += '_Clients_Fat' + str(fat_clients_ratio) +'_Thin_' + str(thin_clients_ratio)
results_file_name += '_Data_Fat_' + str(fat_train_data_ratio) +'_Thin_' + str(thin_train_data_ratio) + '.csv'

results_csv = open(results_file_name, 'w')
results_writer = csv.writer(results_csv, delimiter=',')

results_writer.writerow(['', '', '', '', 'Mean', '', '', '', ''])
results_writer.writerow(['N', 'R', 'Opt', 'Rnd', 'Prp_r2=1', 'Prp_r2=2', 'Prp_r2=3', 'Prp_r2=4', 'Prp_r2=5'])

for r_co in range(no_Rs):
    results_writer.writerow([n_clients, active_clients_list[r_co], \
                            confid_list[r_co][0][0], confid_list[r_co][1][0], confid_list[r_co][2][0], confid_list[r_co][3][0], 
                            confid_list[r_co][4][0], confid_list[r_co][5][0], confid_list[r_co][6][0]])

results_writer.writerow('')
results_writer.writerow(['', '', '', '', 'Lower', '', '', '', ''])
results_writer.writerow(['N', 'R', 'Opt', 'Rnd', 'Prp_r2=1', 'Prp_r2=2', 'Prp_r2=3', 'Prp_r2=4', 'Prp_r2=5'])
for r_co in range(no_Rs):
    results_writer.writerow([n_clients, active_clients_list[r_co], \
                            confid_list[r_co][0][1], confid_list[r_co][1][1], confid_list[r_co][2][1], confid_list[r_co][3][1], 
                            confid_list[r_co][4][1], confid_list[r_co][5][1], confid_list[r_co][6][1]])

results_writer.writerow('')
results_writer.writerow(['', '', '', '', 'Upper', '', '', '', ''])
results_writer.writerow(['N', 'R', 'Opt', 'Rnd', 'Prp_r2=1', 'Prp_r2=2', 'Prp_r2=3', 'Prp_r2=4', 'Prp_r2=5'])
for r_co in range(no_Rs):
    results_writer.writerow([n_clients, active_clients_list[r_co], \
                            confid_list[r_co][0][2], confid_list[r_co][1][2], confid_list[r_co][2][2], confid_list[r_co][3][2], 
                            confid_list[r_co][4][2], confid_list[r_co][5][2], confid_list[r_co][6][2]])

results_csv.close()
