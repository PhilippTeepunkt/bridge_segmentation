#!/usr/bin/python3

import matplotlib.pyplot as plt

labels = ["Elstertal", "Scherkonde", "Unstruttal", "Truckenthal", "Hochmosel", "Koernetal", "Auetal"]
f_measures_opt = [0.9308, 0.6311, 0.5293,  0.3643, 0.5541, 0.2096, 0.6862]
f_measures_avg = [0.9159, 0.5650, 0.4535, 0.1178, 0.5130, 0.0548, 0]
f_distance = [] 
width = 0.35

if __name__ == "__main__":

	for index, f in enumerate(f_measures_opt):
		f_distance.append(float(f) - float(f_measures_avg[index]))

	print(f_distance)

	fig, ax = plt.subplots(figsize = (8,5))
	ax.bar(labels, f_measures_avg, width, label='default parameter')
	ax.bar(labels, f_distance, width, bottom= f_measures_avg, label ="distance to optimal")
	ax.set_ylabel('F-Score')
	ax.set_title('Comparison default and optimal parameter')
	ax.legend()

	plt.savefig('C:\\Users\\phili\\Desktop\\parameter_default_optimal.png', dpi=1200)

	plt.show()