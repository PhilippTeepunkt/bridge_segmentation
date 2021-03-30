#!/usr/bin/python3

import glob
from shutil import copyfile
import sys
import os
import time
import concurrent.futures
import evaluate
import time

sampling_densitys = 2000000

#slab extraction
neighbouhood_sizes = [550]
smoothness_thresholds = [0.60]
curvature_thresholds = [0.025]
#residuals_thresholds = [0.001]

#filtering parameters
num_cluster = [5] #[2, 3, 4, 5, 6]
point_neighbourhood = [300]#[200, 300, 400, 500]
std_deviation = [0.7] #[0.8, 0.7, 0.6, 0.5]

#bridge_files = []
config_files = []
pipeline_program = ""

def generate_configs():
	index = 0
	for size in neighbouhood_sizes:
		for sth in smoothness_thresholds:
			for cth in curvature_thresholds:
				config = (sampling_densitys,size,sth,cth,num_cluster[0],point_neighbourhood[0], std_deviation[0])
				filename = "./pipeline_configs/"+str(index)+"_config_"+str(size)+"_"+str(sth)+"_"+str(cth)+"_default.txt"
				config_file = open(filename,"w")
				for value in config:
					config_file.write(str(value))
					config_file.write("\n")
				config_file.close()
				config_files.append(filename)
				index = index+1
	print(config_files)

def evaluate_config(pipeline_program, config, config_index, bridge_files):

	for bridge_file in bridge_files:
		abs_path = os.path.abspath(config)
		bridge_name = os.path.splitext(os.path.basename(bridge_file))[0].split("_")[0]
		args = [pipeline_program, bridge_file,"./ground_truth/"+bridge_name+"/"+bridge_name+"_slab_GT.txt","./ground_truth/"+bridge_name+"/"+bridge_name+"_bridge_GT.txt",config_index, abs_path]
		evaluate.main(args)


def search_optimal_config(bridge_name):

	print("\n////////////////// GET BEST CONFIG FOR "+bridge_name+" ///////////////////////////////////////\n")
	evaluation_files = glob.glob("./evaluation_data/"+bridge_name+"/*.txt")
	fscores = []
	conf_numbers = []
	for efile in evaluation_files:

		split = os.path.splitext(os.path.basename(efile))[0].split("_")
		config_number = split[len(split)-1]
		conf_numbers.append(config_number)

		f = open(efile, "r")
		lines = f.readlines()
		fscore = float(lines[len(lines)-2].split()[2])
		fscores.append(fscore)
		f.close()

	best_config = evaluation_files[fscores.index(max(fscores))]
	best_config_index = str(conf_numbers[fscores.index(max(fscores))])
	print("For bridge "+bridge_name+", the "+best_config_index+" config is the one with the hightest f-score of "+str(max(fscores))+".")
	copyfile(best_config, "./parameter_search/"+bridge_name+"_best_config_"+best_config_index+".txt")


if __name__ == "__main__":

	start = time.time()
	if len(sys.argv)<3 :
		print("./search_parameter <pipeline_program> <pcl_file_dir>")
		sys.exit()
	# generate configurations to query
	generate_configs()

	pipeline_program = sys.argv[1]

	# query pcl files
	pcl_file_dir = sys.argv[2]
	files = glob.glob(pcl_file_dir+"/*.pcd")
	# cuncurrency
	threadnum = os.cpu_count()

	print("Execute in parallel with "+str(threadnum)+" threads.")
	futures = []
	pool = concurrent.futures.ProcessPoolExecutor(max_workers = threadnum)

	# evaluate each config for bridges
	for idx, config in enumerate(config_files):
		futures.append(pool.submit(evaluate_config, pipeline_program, config, idx, files))
		#evaluate_config(config, idx, files)

	while not concurrent.futures.wait(futures):
		pass

	search_optimal_config("Elstertal")

	# get best configs
	#for bridge_file in files:
	#	bridge_name = os.path.splitext(os.path.basename(bridge_file))[0].split("_")[0]
	#	futures.append(pool.submit(search_optimal_config, bridge_name))

	#while not concurrent.futures.wait(futures):
	#	pass

	end = time.time()
	print("\n Elapsed Time: "+str(end-start))
	print("\n\ndone.")



