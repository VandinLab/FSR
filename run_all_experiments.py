import os
from tqdm import tqdm

numruns = 10
trials_wy = 1000
trials_fsr = 10
k_results = 10000
run_fsr = 1
run_unionbound = 1
run_wy = 1
numcores = 30

max_card_db = {
    "data/adult.csv": 5 ,
    "data/mushroom.data": 5 ,
    "data/bank-additional-full.csv": 3 ,
    "data/theorem-prover.csv": 3 ,
    "data/abalone.data": 5 ,
    "data/TCGA-brain-cancer.data": 5,
    "data/covtype.data": 3 ,
    "data/gisette.data": 2 ,
    "data/HIGGS_": 3 ,
    "data/SUSY.csv": 3 ,
    "data/kdd-cup.data": 2,
    "data/cancer-rna-seq-merge.csv": 2
    }

# list of information for each dataset
# each entry is a tuple that contains:
# 1: the path to the dataset file
# 2: 1 if the file has an header, 0 otherwise
# 3: name of the target column (integer id if no header in file)
# 4: value of the target column to assign 1 in the binary target value
# 5: 1 if features are all categorical, 0 to assign automatically depending on feature values

db_infos = [ ("data/mushroom.data","0","0","p","1"), ("data/adult.csv","1","salary","high50K","0"), ("data/bank-additional-full.csv","1","y","yes","0"), ("data/theorem-prover.csv","0","51","1","0"), ("data/abalone.data","0","8","1","0"), ("data/TCGA-brain-cancer.data","1","Grade","GBM","0"), ("data/covtype.data","0","54","1","0"), ("data/gisette.data","0","5000","1","0"), ("data/HIGGS_","0","0","1","0"), ("data/SUSY.csv","0","0","1","0"), ("data/kdd-cup.data","1","TARGET_B", "1","0"), ("data/cancer-rna-seq-merge.csv","1","Class", "LUAD","2")]
#db_infos = [ ("data/mnist.csv", "0", "0", "1", "0") ]

cmds = []

for test_ in range(numruns):
    for cond_flag in [0,1]:
        for db_info in db_infos:
            db_path = db_info[0]
            db_header = db_info[1]
            db_target_col_name = db_info[2]
            db_target_col_value_1 = db_info[3]
            db_categorical_features = db_info[4]
            max_l = max_card_db[db_path]
            if run_fsr == 1:
                # run FSR
                numt = trials_fsr
                cmd = "python fsr-alg.py -db "+db_path+" -maxd "+str(max_l)+" -res "+str(numt)+" -head "+db_header+" -target "+db_target_col_name+" -tval "+db_target_col_value_1+" -cat "+db_categorical_features+" -cond "+str(cond_flag)+" -k "+str(k_results)+" -pn "+str(numcores)
                print(cmd)
                cmds.append(cmd)
                #os.system(cmd)
            if cond_flag == 0 and run_unionbound == 1:
                # run union bound
                cmd = "python fsr-alg.py -db "+db_path+" -maxd "+str(max_l)+" -res 1 -head "+db_header+" -target "+db_target_col_name+" -tval "+db_target_col_value_1+" -cat "+db_categorical_features+" -cond 0 -ub 1 -k "+str(k_results)+" -pn "+str(numcores)
                print(cmd)
                cmds.append(cmd)
                #os.system(cmd)
            if cond_flag == 1 and run_wy == 1:
                # run TopKWY
                numt = trials_wy
                cmd = "python fsr-alg.py -db "+db_path+" -maxd "+str(max_l)+" -res "+str(numt)+" -head "+db_header+" -target "+db_target_col_name+" -tval "+db_target_col_value_1+" -cat "+db_categorical_features+" -cond 1 -wy 1 -k "+str(k_results)+" -pn "+str(numcores)
                print(cmd)
                cmds.append(cmd)
                #os.system(cmd)
for cmd in tqdm(cmds):
    print()
    print(cmd)
    print()
    os.system(cmd)
