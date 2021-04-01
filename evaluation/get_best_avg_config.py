#!/usr/bin/python3
import glob
import sys
import os
import statistics

def config_sort(file):
	index = os.path.splitext(os.path.basename(file))[0].split("_")[0]
	return int(index)

def result_sort(file):
	splits = os.path.splitext(os.path.basename(file))[0].split("_")
	index = splits[len(splits)-1]
	return int(index)


if __name__ == "__main__":

	config_files = glob.glob("./pipeline_configs/*.txt")
	config_files.sort(key=config_sort)
	bridge_folders = [name for name in os.listdir("./evaluation_data") if os.path.isdir(os.path.join("./evaluation_data", name))]

	avg_recall = []
	evaluation_files = {}
	for bridge_folder in bridge_folders:
		files = glob.glob("./evaluation_data/"+bridge_folder+"/*.txt")
		files.sort(key=result_sort)
		evaluation_files[bridge_folder] = files

	for index, config in enumerate(config_files):
		recalls = []
		skip = False
		for bridge_folder in bridge_folders:
			ev_files = evaluation_files[bridge_folder]
			ev_files.sort(key=result_sort)
			ef = ev_files[index]
			splits = os.path.splitext(os.path.basename(ef))[0].split("_")
			if int(splits[len(splits)-1]) != index:
				print(splits[len(splits)-1])
				skip = True
				evaluation_files[bridge_folder].insert(index,"")
				print(evaluation_files[index+1])
				print("Added empty element at "+str(index))
				continue

			f = open(ef, "r")
			lines = f.readlines()
			recall = float(lines[3].split()[2])
			recalls.append(recall)
			f.close()
		if skip:
			continue
		avg = float(statistics.mean(recalls))
		avg_recall.append(avg)

	best_config = config_files[avg_recall.index(max(avg_recall))]
	print("Best config with best avg_recall: "+str(best_config))
	print("Recall:"+str(max(avg_recall)))



