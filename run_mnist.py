import os
os.system("python fsr-alg.py -db data/mnist.csv -maxd 1 -res 10 -head 0 -target 0 -tval 1 -cat 0 -cond 1 -k 10000 -ores subgroups-mnist-1.csv")
os.system("python mnist-process-results.py")
