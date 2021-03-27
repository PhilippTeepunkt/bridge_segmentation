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
neighbouhood_sizes = [350, 400, 450, 500, 550, 600, 700]
smoothness_thresholds = [0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
curvature_thresholds = [0.025, 0.03, 0.035, 0.4, 0.45, 0.5, 0.055, 0.6]
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

def search_optimal_config(bridge, file, config_files, pipeline_program):
	
	print("\n////////////////// GET BEST CONFIG FOR "+bridge+" ///////////////////////////////////////\n")
	for idx, config in enumerate(config_files):
		abs_path = os.path.abspath(config)
		args = [pipeline_program, file,"./ground_truth/"+bridge+"/"+bridge+"_slab_GT.txt","./ground_truth/"+bridge+"/"+bridge+"_bridge_GT.txt",idx, abs_path]
		evaluate.main(args)

	evaluation_files = glob.glob("./evaluation_data/"+bridge+"/*.txt")
	fscores = []
	for efile in evaluation_files:
		f = open(efile, "r")
		lines = f.readlines()
		f_score = lines[len(lines)-2].split()[2]
		fscores.append(f_score)
		f.close()

	best_config = evaluation_files[fscores.index(max(fscores))]

	print("For bridge "+bridge+", the "+ str(fscores.index(max(fscores)))+" config is the one with the hightest f-score of "+str(f_score)+".")
	print("\n"+best_config)
	print("\n"+str(fscores.index(max(fscores))))
	copyfile(best_config,"./parameter_search/"+bridge+"_best_config_"+str(fscores.index(max(fscores)))+".txt")

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
	print(files)

	# cuncurrency
	threadnum = os.cpu_count()

	#TEMP
	threadnum = 2
	
	print("Execute in parallel with "+str(threadnum)+" threads.")
	futures = []
	pool = concurrent.futures.ProcessPoolExecutor(max_workers = threadnum)
	
	# determine best config per bridge
	for file in files:
		bridge_name = os.path.splitext(os.path.basename(file))[0].split("_")[0]
		futures.append(pool.submit(search_optimal_config, bridge_name, file, config_files, pipeline_program))

	while not concurrent.futures.wait(futures):
		pass

	end = time.time()
	print("\n Elapsed Time: "+str(end-start))
	print("\n\ndone.")



