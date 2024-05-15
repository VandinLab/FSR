import os
from tqdm import tqdm

numruns = 10
trials = [1 , 5 , 10 , 25 , 50 , 100 , 250]
k_results = 10000
run_fsr = 1
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

db_infos = [ ("data/mushroom.data","0","0","p","1"), ("data/adult.csv","1","salary","high50K","0"), ("data/abalone.data","0","8","1","0") ]

cmds = []

for test_ in range(numruns):
    for cond_flag in [0,1]:
        for num_trial in trials:
            for db_info in db_infos:
                db_path = db_info[0]
                db_header = db_info[1]
                db_target_col_name = db_info[2]
                db_target_col_value_1 = db_info[3]
                db_categorical_features = db_info[4]
                max_l = max_card_db[db_path]
                if run_fsr == 1:
                    # run FSR
                    numt = num_trial
                    cmd = "python fsr-alg.py -db "+db_path+" -maxd "+str(max_l)+" -res "+str(numt)+" -head "+db_header+" -target "+db_target_col_name+" -tval "+db_target_col_value_1+" -cat "+db_categorical_features+" -cond "+str(cond_flag)+" -k "+str(k_results)+" -o results_params.csv -pn "+str(numcores)
                    print(cmd)
                    cmds.append(cmd)
                    #os.system(cmd)

for cmd in tqdm(cmds):
    print()
    print(cmd)
    print()
    os.system(cmd)
