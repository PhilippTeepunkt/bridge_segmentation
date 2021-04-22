#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ["Elstertal", "Scherkonde", "Unstruttal", "Truckenthal", "Hochmosel", "Koernetal", "Auetal"]

precisions_slab = [0.89, 0.61, 0.42, 0.06, 0.07, 0.14, 0.0]
recalls_slab = [0.93, 0.52, 0.48, 0.65,0.38, 0.03, 0.15]

precisions = [0.96, 0.85, 0.83, 0.25, 0.74, 0.14, 0.79]
recalls = [0.83, 0.53, 0.49, 0.81, 0.27, 0.03, 0.15]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig1, ax1 = plt.subplots(figsize = (8,5))
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
plt.savefig('C:\\Users\\phili\\Desktop\\evaluation_recall_precision_slab.png', dpi=1000)

fig2, ax2 = plt.subplots(figsize = (8,5))
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

matplotlib.rc('axes', labelsize=10)
#matplotlib.rc('xtick', labelsize=10)
#matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('axes', titlesize=10)
matplotlib.rc('legend', fontsize=10)

plt.savefig('C:\\Users\\phili\\Desktop\\evaluation_recall_precision_bridge.png', dpi=1000)
plt.show()