#!/usr/bin/python3
import glob
import sys
import os

if __name__ == "__main__":

	# query pcl files
	result_file_dir = sys.argv[1]
	files = glob.glob(result_file_dir+"/*.txt")

	f_scores = []
	for result in files:
		bridge_name = os.path.splitext(os.path.basename(result))[0].split("_")[0]
		#print (bridge_name)

		f = open(result, "r")
		lines = f.readlines()
		f_score = lines[len(lines)-2].split()[2]
		print(lines[len(linse)-2])
		f_scores.append(f_score)
		f.close()
	print("Best f-score:"+str(max(f_scores)))
	print("Config:"+str(f_scores.index(max(f_scores))))






