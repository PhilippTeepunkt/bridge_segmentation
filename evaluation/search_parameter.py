#!/usr/bin/python3

import glob
from shutil import copyfile
import sys
import os
import time
import subprocess
import evaluate

sampling_sizes = [400, 500, 600, 700]
smoothness_thresholds = [0.62]
curvature_thresholds = [1.6]

num_cluster_first = [4]
num_cluster_second = [3]
point_neighbourhood = [200]
std_deviation = [0.8]

#bridge_files = []
config_files = []
pipeline_program = ""

def generate_configs():
	for size in sampling_sizes:
		for sth in smoothness_thresholds:
			for cth in curvature_thresholds:
				config = (size,sth,cth,num_cluster_first[0],num_cluster_second[0],point_neighbourhood[0], std_deviation[0])
				filename = "./pipeline_configs/Config_"+str(size)+"_"+str(sth)+"_"+str(cth)+"_default.txt"
				config_file = open(filename,"w")
				for value in config:
					config_file.write(str(value))
					config_file.write("\n")
				config_file.close()
				config_files.append(filename)

def search_optimal_config(bridge, file):
	
	print("\n////////////////// GET BEST CONFIG FOR "+bridge+" ///////////////////////////////////////\n")
	for idx, config in enumerate(config_files):
		abs_path = os.path.abspath(config)
		print(abs_path)
		args = [pipeline_program, file,"./ground_truth/"+bridge+"/"+bridge+"_slab_GT.txt","./ground_truth/"+bridge+"/"+bridge+"_bridge_GT.txt", str(idx), abs_path]
		evaluate.main(args)

	evaluation_files = glob.glob("./evaluation_data/"+bridge+"/*.txt")
	fscores = []
	for efile in evaluation_files:
		f = open(efile, "r")
		lines = f.readlines()
		f_score = lines[len(lines)-2].split()[2]
		fscores.append(f_score)
		f.close()

	best_config = fscores.index(max(fscores))

	print("For bridge "+bridge+", the "+str(best_config)+". config is the one with the hightest f-score of "+str(f_score)+".")
	copyfile(evaluation_files[best_config],"./parameter_search/"+bridge+"_best_config_"+str(best_config)+".txt")

if __name__ == "__main__":

	if len(sys.argv)<3 :
		print("./search_parameter <pipeline_program> <pcl_file_dir>")
		sys.exit()
	# generate configurations to query
	generate_configs()

	pipeline_program = sys.argv[1]

	# query pcl files
	pcl_file_dir = sys.argv[2]
	files = glob.glob(pcl_file_dir+"/*.pcd")
	print(files)
	for file in files:
		bridge_name = os.path.splitext(os.path.basename(file))[0].split("_")[0]

		# determine best config per bridge		
		search_optimal_config(bridge_name, file)



