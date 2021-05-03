import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys

file_in1 = sys.argv[1]
file_in2 = sys.argv[2]
file_in3 = sys.argv[3]
file_out = sys.argv[4]
column   = sys.argv[8]

import csv

x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []




with open(file_in1,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    ignore = next(plots)
    for row in plots:
        x1.append(float(row[1]))
        y1.append(float(row[int(column)]))

with open(file_in2,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)    
    ignore = next(plots)
    for row in plots:
        x2.append(float(row[1]))
        y2.append(float(row[int(column)]))

with open(file_in3,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    ignore = next(plots)
    for row in plots:
        x3.append(float(row[1]))
        y3.append(float(row[int(column)]))

xmax = x1[0]
for i in range(len(x1)):
    x1[i] = 100 * x1[i] / xmax

for i in range(len(x2)):
    x2[i] = 100 * x2[i] / xmax

for i in range(len(x3)):
    x3[i] = 100 * x3[i] / xmax

fig = plt.figure()
a1 = fig.add_subplot(1,1,1)
a1.plot(x1, y1, label=sys.argv[8], linewidth=1, fontsize=16) 
a1.tick_params(axis='x', labelsize=14)
a1.tick_params(axis='y', labelsize=12)
a2 = fig.add_subplot(1,1,1)
a2.plot(x2, y2, label=sys.argv[9], linewidth=1, fontsize=16)
a2.tick_params(axis='x', labelsize=14)
a2.tick_params(axis='y', labelsize=12)
a3 = fig.add_subplot(1,1,1)
a3.plot(x3, y3, label=sys.argv[10], linewidth=1, fontsize=16)
a3.tick_params(axis='x', labelsize=14)
a3.tick_params(axis='y', labelsize=12)

plt.xlabel('Pecentage of Parameters Remaining', fontsize=16)
plt.ylabel(sys.argv[6], fontsize=16)
plt.title(sys.argv[5])                                                                                                                       
plt.legend(fontsize=16)
plt.show()
plt.savefig(file_out)

