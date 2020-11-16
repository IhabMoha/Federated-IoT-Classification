import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv


no_comm_rounds = 20
client_no_of_epoches = 8
fat_clients_ratio = 0.20
thin_clients_ratio = 0.80
fat_train_data_ratio = 0.10
thin_train_data_ratio = 0.01

#N_list = [100, 200, 400, 800, 1600]
R_list = [10, 20, 30, 40, 50]

no_runs = 5
no_Ns = 5
no_Rs = 5
no_algs = 7
confid_list = [[[[[] for z in range(3)] for y in range(no_algs)] for x in range(no_Rs)] for n in range(no_Ns)]
avg_acc_list = [[[[] for y in range(no_algs)] for x in range(no_Rs)] for n in range(no_Ns)]
fat_per_list = [[[[] for y in range(no_algs)] for x in range(no_Rs)] for n in range(no_Ns)]

# Load results
#results_file_name = '/home/shadha/Dropbox/Development/IoT/Results/Final'
results_file_name = 'C:\\Users\\14792\\Dropbox\\Development\\IoT\\Results\\Final'
results_file_name += '_Rounds_' + str(no_comm_rounds) +'_Epochs_' + str(client_no_of_epoches)
results_file_name += '_Clients_Fat' + str(fat_clients_ratio) +'_Thin_' + str(thin_clients_ratio)
results_file_name += '_Data_Fat_' + str(fat_train_data_ratio) +'_Thin_' + str(thin_train_data_ratio) + '.csv'

results_csv = open(results_file_name, 'r')
results_reader = csv.reader(results_csv, delimiter=',')

# Skip rows
next(results_reader)
next(results_reader)

# Get Mean of Confidence Intervals
r_co = 0
for row in results_reader:
    #print(row)
    for n_co in range(no_Ns):
        for alg_co in range(no_algs):
            confid_list[n_co][r_co][alg_co][0] = float(row[((n_co+1)*2)+(n_co*no_algs)+alg_co])
    r_co += 1
    if r_co == no_Rs:
        break
# Skip rows
for co in range(3):
    next(results_reader)

# Get Lower of Confidence Intervals
r_co = 0
for row in results_reader:
    #print(row)
    for n_co in range(no_Ns):
        for alg_co in range(no_algs):
            confid_list[n_co][r_co][alg_co][1] = float(row[((n_co+1)*2)+(n_co*no_algs)+alg_co])
    r_co += 1
    if r_co == no_Rs:
        break

# Skip rows
for co in range(3):
    next(results_reader)

# Get Upper of Confidence Intervals
r_co = 0
for row in results_reader:
    #print(row)
    for n_co in range(no_Ns):
        for alg_co in range(no_algs):
            confid_list[n_co][r_co][alg_co][2] = float(row[((n_co+1)*2)+(n_co*no_algs)+alg_co])
    r_co += 1
    if r_co == no_Rs:
        break

# Skip rows
for co in range(3):
    next(results_reader)

# Get Average Accuracy per client
r_co = 0
for row in results_reader:
    #print(row)
    for n_co in range(no_Ns):
        for alg_co in range(no_algs):
            avg_acc_list[n_co][r_co][alg_co] = float(row[((n_co+1)*2)+(n_co*no_algs)+alg_co])
    r_co += 1
    if r_co == no_Rs:
        break

# Skip rows
for co in range(3):
    next(results_reader)

# Get Fat percentage
r_co = 0
for row in results_reader:
    #print(row)
    #tot_fat = 10.0
    for n_co in range(no_Ns):
        for alg_co in range(no_algs):
            fat_per_list[n_co][r_co][alg_co] = float(row[((n_co+1)*2)+(n_co*no_algs)+alg_co])# / tot_fat
    #tot_fat += 10
    r_co += 1
    if r_co == no_Rs:
        break

results_csv.close()


'''
# Display values for checking
for r_co in range(no_Rs):
    for n_co in range(no_Ns):
        for alg_co in range(no_algs):
            print(fat_per_list[n_co][r_co][alg_co], end=' ')
            #print(confid_list[n_co][r_co][alg_co][2], end=' ')
        print(' -N- ', end = '')
    print()
'''

# -------------------------------------------------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------------------------------------------------

font_size = 14
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
#plt.rc('text', usetex=True)


labels = ['Best', 'Rand', r'Prop $r_2$=1', r'Prop $r_2$=2', r'Prop $r_2$=3', r'Prop $r_2$=4', r'Prop $r_2$=5']
colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'black']
hatches = ['/', 'o', 'x', '-', '+', '\\', '.']
x_ticks = ['', '', '1', '2', '3', '4', '5']
barWidth = 0.2
N = 2
R = 1


# ------------------------------------------
# [1,1] Plot Accuracy V.S r2
# ------------------------------------------

# Set bar heights
bars = [[] for x in range(no_algs)]
for alg_co in range(no_algs):
    bars[alg_co] = confid_list[N][R][alg_co][0]#*100
  
# Set heights of err
yerrs = [[] for x in range(no_algs)]
for alg_co in range(no_algs):
    lower = confid_list[N][R][alg_co][0] - confid_list[N][R][alg_co][1]
    upper = confid_list[N][R][alg_co][2] - confid_list[N][R][alg_co][0]
    yerrs[alg_co] = np.array([[lower], [upper]])

# The x position of bars
locs = []
for alg_co in range(no_algs+4):
    locs.append(barWidth+(barWidth+0.1)*alg_co)
 
# Create blue bars
for b_co in range(no_algs):
    plt.bar(locs[b_co], bars[b_co], width = barWidth, color = colors[b_co], edgecolor = 'black',\
            yerr=yerrs[b_co], capsize=7, label=labels[b_co], hatch = hatches[b_co])
  
# general layout
plt.xticks(locs, x_ticks)
plt.ylabel('Test Accuracy')
plt.xlabel(r'$r_2$')
plt.legend(loc='center right')
#xmin, xmax, ymin, ymax
plt.axis([0, 3.4, 0.50, 0.82])

# Show graphic
plt.show()


# ------------------------------------------
# [1,2] Plot Average Accuracy per client V.S r2
# ------------------------------------------

# Set bar heights
bars = [[] for x in range(no_algs)]
for alg_co in range(no_algs):
    bars[alg_co] = avg_acc_list[N][R][alg_co]#*100
  
# The x position of bars
locs = []
for alg_co in range(no_algs+4):
    locs.append(barWidth+(barWidth+0.1)*alg_co)
 
# Create blue bars
for b_co in range(no_algs):
    plt.bar(locs[b_co], bars[b_co], width = barWidth, color = colors[b_co], edgecolor = 'black', label=labels[b_co], hatch=hatches[b_co])
    #break
  
# general layout
plt.xticks(locs, x_ticks)
plt.ylabel('Average Accuracy Per Selected Client')
plt.xlabel(r'$r_2$')
plt.legend(loc='center right')
plt.axis([0, 3.4, 0.0, 0.65])
# Show graphic
plt.show()


# ------------------------------------------
# [1,3] Plot Fat Percentage V.S r2
# ------------------------------------------

# Set bar heights
bars = [[] for x in range(no_algs)]
for alg_co in range(no_algs):
    bars[alg_co] = fat_per_list[N][R][alg_co] / R_list[R]
print(bars)
  
# The x position of bars
locs = []
for alg_co in range(no_algs+4):
    locs.append(barWidth+(barWidth+0.1)*alg_co)
 
# Create blue bars
for b_co in range(no_algs):
    plt.bar(locs[b_co], bars[b_co], width = barWidth, color = colors[b_co], edgecolor = 'black', label=labels[b_co], hatch=hatches[b_co])
    #break
  
# general layout
plt.xticks(locs, x_ticks)
plt.ylabel('% Fat Clients')
plt.xlabel(r'$r_2$')
plt.legend(loc='center right')
plt.axis([0, 3.4, 0.0, 1.05])
# Show graphic
plt.show()


r2 = 2+3
labels = ['Best', 'Rand', 'Prop']
x_ticks = ['10', '20', '30', '40', '50']
colors = ['red', 'green', 'cyan']
hatches = ['/', 'o', '\\']


# ------------------------------------------
# [2,1] Plot Accuracy V.S R
# ------------------------------------------

# Set bar heights
bars = [[[] for x in range(no_Rs)] for y in range(3)]
for r_co in range(no_Rs):
    bars[0][r_co] = confid_list[N][r_co][0][0]
    bars[1][r_co] = confid_list[N][r_co][1][0]
    bars[2][r_co] = confid_list[N][r_co][r2][0]

# Set heights of err
yerrs = [[np.array([]), np.array([])] for y in range(3)]
for r_co in range(no_Rs):
    lower = confid_list[N][r_co][0][0] - confid_list[N][r_co][0][1]
    upper = confid_list[N][r_co][0][2] - confid_list[N][r_co][0][0]
    yerrs[0][0] = np.append(yerrs[0][0], lower)
    yerrs[0][1] = np.append(yerrs[0][1], upper)
    
    lower = confid_list[N][r_co][1][0] - confid_list[N][r_co][1][1]
    upper = confid_list[N][r_co][1][2] - confid_list[N][r_co][1][0]
    yerrs[1][0] = np.append(yerrs[1][0], lower)
    yerrs[1][1] = np.append(yerrs[1][1], upper)
    
    lower = confid_list[N][r_co][r2][0] - confid_list[N][r_co][r2][1]
    upper = confid_list[N][r_co][r2][2] - confid_list[N][r_co][r2][0]
    yerrs[2][0] = np.append(yerrs[2][0], lower)
    yerrs[2][1] = np.append(yerrs[2][1], upper)
    
# Create blue bars
x = 0.2
for b_co in range(3):
    # The x position of bars
    locs = []
    for alg_co in range(no_Rs):
        locs.append(x+alg_co*4*barWidth)
    plt.bar(locs, bars[b_co], width = barWidth, color = colors[b_co], edgecolor = 'black',\
            yerr=[yerrs[b_co][0], yerrs[b_co][1]], capsize=7, label=labels[b_co], hatch = hatches[b_co])
    x += barWidth
  
# general layout
locs = []
for alg_co in range(no_Rs):
    locs.append(0.4+alg_co*(barWidth*4))
plt.xticks(locs, x_ticks)
plt.ylabel('Test Accuracy')
plt.xlabel(r'Number of Selected Clients ($R$)')
plt.legend(loc='upper center', ncol=3)
plt.axis([0, 4, 0.5, 0.92])
# Show graphic
plt.show()

# ------------------------------------------
# [2,2] Plot Average Accuracy per client V.S R
# ------------------------------------------

# Set bar heights
bars = [[[] for x in range(no_Rs)] for y in range(3)]
for r_co in range(no_Rs):
    bars[0][r_co] = avg_acc_list[N][r_co][0]
    bars[1][r_co] = avg_acc_list[N][r_co][1]
    bars[2][r_co] = avg_acc_list[N][r_co][r2]

# Create blue bars
x = 0.2
for b_co in range(3):
    # The x position of bars
    locs = []
    for alg_co in range(no_Rs):
        locs.append(x+alg_co*4*barWidth)
    plt.bar(locs, bars[b_co], width = barWidth, color = colors[b_co], edgecolor = 'black', label=labels[b_co], hatch = hatches[b_co])
    x += barWidth
  
# general layout
locs = []
for alg_co in range(no_Rs):
    locs.append(0.5+alg_co*(barWidth*4))
plt.xticks(locs, x_ticks)
plt.ylabel('Average Accuracy Per Selected Client')
plt.xlabel(r'Number of Selected Clients ($R$)')
plt.legend(loc='upper center', ncol=3)
plt.axis([0, 4, 0.0, 0.75])
# Show graphic
plt.show()

# ------------------------------------------
# [2,3] Plot Percentage of fat clients V.S R
# ------------------------------------------

# Set bar heights
bars = [[[] for x in range(no_Rs)] for y in range(3)]
for r_co in range(no_Rs):
    bars[0][r_co] = fat_per_list[N][r_co][0] / R_list[r_co]
    bars[1][r_co] = fat_per_list[N][r_co][1] / R_list[r_co]
    bars[2][r_co] = fat_per_list[N][r_co][r2] / R_list[r_co]

# Create blue bars
x = 0.2
for b_co in range(3):
    # The x position of bars
    locs = []
    for alg_co in range(no_Rs):
        locs.append(x+alg_co*4*barWidth)
    plt.bar(locs, bars[b_co], width = barWidth, color = colors[b_co], edgecolor = 'black', label=labels[b_co], hatch = hatches[b_co])
    x += barWidth
  
# general layout
locs = []
for alg_co in range(no_Rs):
    locs.append(0.4+alg_co*(barWidth*4))
plt.xticks(locs, x_ticks)
plt.ylabel('% Fat Clients')
plt.xlabel(r'Number of Selected Clients ($R$)')
plt.legend(loc='upper center', ncol=3)
plt.axis([0, 4, 0.0, 1.15])
# Show graphic
plt.show()


x_ticks = ['100', '200', '400', '800', '1600']


# ------------------------------------------
# [3,1] Plot Accuracy V.S N
# ------------------------------------------

# Set bar heights
bars = [[[] for x in range(no_Ns)] for y in range(3)]
for n_co in range(no_Ns):
    bars[0][n_co] = confid_list[n_co][R][0][0]
    bars[1][n_co] = confid_list[n_co][R][1][0]
    bars[2][n_co] = confid_list[n_co][R][r2][0]

# Set heights of err
yerrs = [[np.array([]), np.array([])] for y in range(3)]
for n_co in range(no_Ns):
    lower = confid_list[n_co][R][0][0] - confid_list[n_co][R][0][1]
    upper = confid_list[n_co][R][0][2] - confid_list[n_co][R][0][0]
    yerrs[0][0] = np.append(yerrs[0][0], lower)
    yerrs[0][1] = np.append(yerrs[0][1], upper)
    
    lower = confid_list[n_co][R][1][0] - confid_list[n_co][R][1][1]
    upper = confid_list[n_co][R][1][2] - confid_list[n_co][R][1][0]
    yerrs[1][0] = np.append(yerrs[1][0], lower)
    yerrs[1][1] = np.append(yerrs[1][1], upper)
    
    lower = confid_list[n_co][R][r2][0] - confid_list[n_co][R][r2][1]
    upper = confid_list[n_co][R][r2][2] - confid_list[n_co][R][r2][0]
    yerrs[2][0] = np.append(yerrs[2][0], lower)
    yerrs[2][1] = np.append(yerrs[2][1], upper)
 
# Create blue bars
x = 0.2
for b_co in range(3):
    # The x position of bars
    locs = []
    for alg_co in range(no_Rs):
        locs.append(x+alg_co*4*barWidth)
    plt.bar(locs, bars[b_co], width = barWidth, color = colors[b_co], edgecolor = 'black',\
            yerr=yerrs[b_co], capsize=7, label=labels[b_co], hatch = hatches[b_co])
    x += barWidth
  
# general layout
locs = []
for alg_co in range(no_Rs):
    locs.append(0.4+alg_co*(barWidth*4))
plt.xticks(locs, x_ticks)
plt.ylabel('Test Accuracy')
plt.xlabel(r'Total Number of Clients ($N$)')
plt.legend(loc='upper center', ncol=3)
plt.axis([0, 4, 0.5, 0.91])
# Show graphic
plt.show()


# ------------------------------------------
# [3,2] Plot Average Accuracy per client V.S N
# ------------------------------------------

# Set bar heights
bars = [[[] for x in range(no_Ns)] for y in range(3)]
for n_co in range(no_Ns):
    bars[0][n_co] = avg_acc_list[n_co][R][0]
    bars[1][n_co] = avg_acc_list[n_co][R][1]
    bars[2][n_co] = avg_acc_list[n_co][R][r2]

# Create blue bars
x = 0.2
for b_co in range(3):
    # The x position of bars
    locs = []
    for alg_co in range(no_Rs):
        locs.append(x+alg_co*4*barWidth)
    plt.bar(locs, bars[b_co], width = barWidth, color = colors[b_co], edgecolor = 'black', label=labels[b_co], hatch = hatches[b_co])
    x += barWidth
  
# general layout
locs = []
for alg_co in range(no_Rs):
    locs.append(0.4+alg_co*(barWidth*4))
plt.xticks(locs, x_ticks)
plt.ylabel('Average Accuracy Per Selected Client')
plt.xlabel(r'Total Number of Clients ($N$)')
plt.legend(loc='upper center', ncol=3)
plt.axis([0, 4, 0.0, 0.75])
# Show graphic
plt.show()

# ------------------------------------------
# [3,3] Plot Percentage of Fat Clients V.S N
# ------------------------------------------

# Set bar heights
bars = [[[] for x in range(no_Ns)] for y in range(3)]
for n_co in range(no_Ns):
    bars[0][n_co] = fat_per_list[n_co][R][0] / R_list[R]
    bars[1][n_co] = fat_per_list[n_co][R][1] / R_list[R]
    bars[2][n_co] = fat_per_list[n_co][R][r2] / R_list[R]

# Create blue bars
x = 0.2
for b_co in range(3):
    # The x position of bars
    locs = []
    for alg_co in range(no_Rs):
        locs.append(x+alg_co*4*barWidth)
    plt.bar(locs, bars[b_co], width = barWidth, color = colors[b_co], edgecolor = 'black', label=labels[b_co], hatch = hatches[b_co])
    x += barWidth
  
# general layout
locs = []
for alg_co in range(no_Rs):
    locs.append(0.4+alg_co*(barWidth*4))
plt.xticks(locs, x_ticks)
plt.ylabel('% Fat Clients')
plt.xlabel(r'Total Number of Clients ($N$)')
plt.legend(loc='upper center', ncol=3)
plt.axis([0, 4, 0.0, 1.15])
# Show graphic
plt.show()



# ------------------------------------------------------------------------------------------------------------------------------------------
'''

# ------------------------------------------
# Plot probabilities v.s. \alpha
# ------------------------------------------

labels = [r'$r_1$=2,$r_2$=5', r'$r_1$=3,$r_2$=5', r'$r_1$=3,$r_2$=7', r'$r_1$=5,$r_2$=7']
colors = ['green', 'red', 'blue', 'black']
hatches = ['/', 'o', 'x', '-', '+', '\\', '.']
#x_ticks = ['', '', '1', '2', '3', '4', '5']
barWidth = 0.2
N = 1000

x = np.linspace(0, 1000, 11)


def betweenr1_r2_probability(r1,r2,alpha,N):
    pr = []
    mm=len(alpha)
    for k in range(0, mm):
        x=alpha[k]/N
        S=0
        for j in range(r1, r2+1):
            S=S+np.math.log(1/x)**j/np.math.factorial(j)
        pr.append(x * S)
    return pr


alpha=[x for x in range(1,N+1)]
plt.plot(alpha, betweenr1_r2_probability(2,5,alpha,N), 'g:', alpha,betweenr1_r2_probability(3,5,alpha,N), 'r-', alpha,betweenr1_r2_probability(3,7,alpha,N), 'b--', alpha,betweenr1_r2_probability(5,7,alpha,N),'k-.')


# The x position of bars
#locs = []
#for alg_co in range(no_algs+4):
#    locs.append(barWidth+(barWidth+0.1)*alg_co)
 
# general layout
#plt.xticks(locs, x_ticks)
plt.ylabel('Total probability')
plt.xlabel(r'$\alpha$')
plt.legend(labels, loc='center right')
plt.axis([0, 1000, 0, 0.8])

# Show graphic
plt.show()
'''