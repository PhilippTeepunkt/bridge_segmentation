#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ["Elstertal", "Scherkonde", "Unstruttal", "Truckenthal", "Hochmosel", "Koernetal", "Auetal"]

precisions_slab = [0.8955, 0.6144, 0.4249, 0.0647, 0.07674, 0.1426, 0.0]
recalls_slab = [0.9372, 0.5229, 0.4863, 0.6542,0.3853, 0.0339, 0.1574]

precisions = [0.9642, 0.8585, 0.8397, 0.2548, 0.7480, 0.1426, 0.7976]
recalls = [0.8305, 0.5328, 0.4900, 0.8132, 0.2751, 0.0311, 0.1513]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig1, ax1 = plt.subplots()
rects1 = ax1.bar(x - width/2, precisions_slab, width, label='Precision')
rects2 = ax1.bar(x + width/2, recalls_slab, width, label='Recall')

ax1.set_ylabel('Value')
ax1.set_title('Slab Extraction Results')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

def autolabel(rects,axis):
	"""Attach a text label above each bar in *rects*, displaying its height."""
	for rect in rects:
		height = rect.get_height()
		axis.annotate('{}'.format(height),
					xy=(rect.get_x() + rect.get_width() / 2, height),
					xytext=(0, 3),  # 3 points vertical offset
					textcoords="offset points",
					ha='center', va='bottom')

autolabel(rects1,ax1)
autolabel(rects2,ax1)

fig1.tight_layout()

fig2, ax2 = plt.subplots()
rects3 = ax2.bar(x - width/2, precisions, width, label='Precision')
rects4 = ax2.bar(x + width/2, recalls, width, label='Recall')

ax2.set_ylabel('Value')
ax2.set_title('Bridge Extraction Results')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()

autolabel(rects3, ax2)
autolabel(rects4, ax2)

fig2.tight_layout()

plt.show()