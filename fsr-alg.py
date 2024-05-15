import math
import os
import time
import pysubgroup as ps
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import scipy
from scipy.special import comb
from scipy.stats import bernoulli
from scipy.stats import binom
parser = argparse.ArgumentParser()
parser.add_argument("-db", help="path to input file")
parser.add_argument("-target", help="string of target column")
parser.add_argument("-tval", help="value of target to consider")
parser.add_argument("-k", help="number of top-k results to find (def.=10000)", type=int, default = 10000)
parser.add_argument("-maxd", help="max number of conjunction terms (def.=3)", type=int, default = 3)
parser.add_argument("-eps", help="default value of eps to use for output (ignored when corr=1)", type=float, default = 0.01)
parser.add_argument("-corr", help="run correction (def.=1)", type=int, default = 1)
parser.add_argument("-mine", help="run mining (def.=1)", type=int, default = 1)
parser.add_argument("-res", help="number of resamples (def.=10)", type=int, default = 10)
parser.add_argument("-p", help="parallel computations (def.=1)",default=1,type=int)
parser.add_argument("-pn", help="number of parallel cores (def.=0 use all)",default=0,type=int)
parser.add_argument("-dfs", help="use dfs exploration (def.=1)",default=1,type=int)
parser.add_argument("-simp", help="use simple exploration (def.=0)",default=0,type=int)
parser.add_argument("-wy", help="run WY correction (def.=0)",default=0,type=int)
parser.add_argument("-ub", help="use union bound (def.=0)",default=0,type=int)
parser.add_argument("-cond", help="use conditional distribution correction (def.=0)",default=0,type=int)
parser.add_argument("-o", help="output path",default="results_signfsr.csv")
parser.add_argument("-ores", help="output path for significant subgroups (def. no output)",default="")
parser.add_argument("-d", help="delta (def. 0.05)",default=0.05, type=float)
parser.add_argument("-cat", help="categorize data (def. 0)",default=0, type=int)
parser.add_argument("-geneexp", help="custom search space for gene expression data (def. 0)",default=0, type=int)
parser.add_argument("-head", help="1 if data has an header (def. 0)",default=0, type=int)
args = parser.parse_args()

debug_ = False

class MyQualityFunction(ps.StandardQF):

    def __init__(self , mu_fixed_):
        self.mu_fixed = mu_fixed_
        super().__init__(1.0)


    @staticmethod
    def standard_qf(a, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup, mu_fixed_):
        if not hasattr(instances_subgroup, '__array_interface__') and (instances_subgroup == 0):
            return np.nan
        negatives_subgroup = instances_subgroup - positives_subgroup
        quality_fixed_mu_ = (positives_subgroup*(1-mu_fixed_) - negatives_subgroup*mu_fixed_)/instances_dataset
        if debug_:
            p_subgroup = np.divide(positives_subgroup, instances_subgroup)
            p_dataset = positives_dataset / instances_dataset
            #return (instances_subgroup / instances_dataset) ** a * (p_subgroup - p_dataset)
            wracc_ = (instances_subgroup / instances_dataset) ** a * (p_subgroup - p_dataset)
            print("call to standard_qf")
            print("    positives_subgroup",positives_subgroup,"instances_subgroup",instances_subgroup)
            print("    positives_dataset",positives_dataset,"instances_dataset",instances_dataset)
            print("    quality_fixed_mu_",quality_fixed_mu_)
            print("    wracc_",wracc_)
        return quality_fixed_mu_

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return MyQualityFunction.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.size_sg, statistics.positives_count, self.mu_fixed)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return MyQualityFunction.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.positives_count, statistics.positives_count, self.mu_fixed)

    def optimistic_generalisation(self, subgroup, target, data, statistics=None):
        print("call to optimistic_generalisation, not implemented!")
        exit()
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        pos_remaining = dataset.positives_count - statistics.positives_count
        return MyQualityFunction.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.size_sg + pos_remaining, dataset.positives_count, self.mu_fixed)




def mine_sd(dataset_ , k_ , target_col_ , max_depth_ , searchspace , fixed_mu_ , min_quality_=0):
    start = time.time()
    target = ps.BinaryTarget (target_col_, 1)

    task = ps.SubgroupDiscoveryTask (
        dataset_,
        target,
        searchspace,
        result_set_size=k_,
        depth=max_depth_,
        min_quality=min_quality_,
        qf=MyQualityFunction(fixed_mu_))
    if args.dfs == 1 and args.simp == 0:
        result = ps.DFS(ps.BitSetRepresentation).execute(task)
    else:
        if args.simp == 1:
            print("using simple search...")
            result = ps.SimpleSearch().execute(task)
        else:
            result = ps.BestFirstSearch().execute(task)
    res_df = result.to_dataframe()
    end = time.time()
    elapsed_ = end - start
    return res_df , elapsed_



# parse the separating character
fin = open(args.db,"r")
line = fin.readline()
if ";" in line:
    sep_char = ";"
else:
    sep_char = ","

if args.head == 1:
    data = pd.read_csv(args.db,sep=sep_char)
else:
    data = pd.read_csv(args.db,header=None,sep=sep_char)
if args.cat == 1:
    data = data.astype('category')
if args.cat == 1:
    args.geneexp = 1
data.columns = data.columns.astype(str)
print(data.columns)
print(data)
num_features = data.shape[1]-1
target_col = args.target
target_val = args.tval
if data.dtypes[target_col] == 'float64':
    data[target_col] = data[target_col].astype(int)
data[target_col] = data[target_col].astype(str)
data[target_col] = (data[target_col] == target_val).astype(int)
target_vals = data[target_col].unique()

print("target_col",target_col)
print("target_val",target_val)
print("target_vals",target_vals)
if len(target_vals) != 2:
    print("the target could not be cast in a binary vector, check again!")
    exit()
#print(data)

#data[target_col] = data[target_col]==target_val
num_samples = data.shape[0]
mu_est = data[target_col].mean()
max_subg_freq_est = 1.
eps_mu = math.sqrt( math.log(4/args.d)/(2*num_samples) )
if args.cond == 1:
    print("Conditional setting!")
    eps_mu = 0.

mu_check_ex_lb = 0.
mu_check_ex_ub = mu_est
k_mu_ = data[target_col].sum()
while mu_check_ex_ub-mu_check_ex_lb > mu_est*0.0001:
    mu_check_ex = (mu_check_ex_ub+mu_check_ex_lb)/2.
    conf_ = 1.-scipy.stats.binom.cdf(k_mu_, num_samples, mu_check_ex)
    if conf_ <= args.d/4:
        mu_check_ex_lb = mu_check_ex
    else:
        mu_check_ex_ub = mu_check_ex


mu_hat_ex = 1.
mu_hat_ex_lb = mu_est
mu_hat_ex_ub = 1.
while mu_hat_ex_ub-mu_hat_ex_lb > mu_est*0.0001:
    mu_hat_ex = (mu_hat_ex_ub+mu_hat_ex_lb)/2.
    conf_ = scipy.stats.binom.cdf(k_mu_, num_samples, mu_hat_ex)
    if conf_ > args.d/4:
        mu_hat_ex_lb = mu_hat_ex
    else:
        mu_hat_ex_ub = mu_hat_ex


mu_hat = min(1,mu_est + eps_mu)
mu_hat = min(mu_hat,mu_hat_ex_ub)
mu_low = max(0,mu_est - eps_mu)
mu_low = max(mu_low,mu_check_ex_lb)
print("num_samples",num_samples)
print("mu_est",mu_est)
if args.cond == 0:
    print("eps_mu",eps_mu)
    print("mu_hat",mu_hat)
    print("mu_low",mu_low)
k_results = args.k
max_depth = args.maxd
eps = args.eps



from multiprocessing import Pool
if args.p == 0:
    workers = Pool(processes=1)
else:
    if args.pn > 0:
        workers = Pool(args.pn)
    else:
        workers = Pool()

use_fixed_mu = 0

max_distinct_values_data = 1
numeric_features = [x for x in data.select_dtypes(include=['number']).columns.values if x not in [target_col]]

# remove constant features
to_remove_features = []
for attr_name in numeric_features:
    if data[attr_name].min() == data[attr_name].max():
        to_remove_features.append(attr_name)
if len(to_remove_features) > 0:
    data.drop(to_remove_features,axis=1,inplace=True)
    numeric_features = [x for x in data.select_dtypes(include=['number']).columns.values if x not in [target_col]]
    print("dropped",len(to_remove_features),"constant features")

non_numeric_features = [x for x in data.columns.values if x not in [target_col] and x not in numeric_features]
print("numeric_features",len(numeric_features))
#print(numeric_features)
print("non_numeric_features",len(non_numeric_features))
for attr_name in numeric_features:
    unique_vals = data[attr_name].unique().shape[0]
    max_distinct_values_data = max(max_distinct_values_data , (unique_vals+2)*(unique_vals+1)/2)
for attr_name in non_numeric_features:
    unique_vals = data[attr_name].unique().shape[0]
    max_distinct_values_data = max(max_distinct_values_data , unique_vals)

def bound_FWER(est_devs_ , mu_ , eps_t_ , num_trials_ , num_samples_):
    num_comb = 0
    count_union_bound = 0
    z = args.maxd
    for i in range(z+1):
        num_comb_i = scipy.special.comb(num_features,i,exact=True)
        num_comb += num_comb_i
        count_union_bound += num_comb_i*(max_distinct_values_data)**i
    bound_pseudo = math.sqrt( (2*z + math.log(num_comb) + math.log(2/args.d))/(2*num_samples) )
    print("num_comb",num_comb)
    print("count_union_bound",count_union_bound)

    if args.cond == 0:
        log_d_ = math.log(4/args.d)
        r_hat = est_devs_ + math.sqrt( log_d_/(2*num_trials_*num_samples_) )
        r_hat_2 = est_devs_ + math.sqrt( 2*max_subg_freq_est*log_d_/(num_trials_*num_samples_) )
        r_hat = min(r_hat,r_hat_2)
        if use_fixed_mu == 1:
            r_hat = 2*r_hat
        mu_hat_ = mu_+eps_t_
        v_ = mu_hat_*(1-mu_hat_)
        if mu_hat_ > 0.5:
            v_ = 0.25
        log_term_ = 2*v_*log_d_/num_samples_
        d_hat = r_hat + math.sqrt( log_term_*log_term_ + 2*r_hat*log_d_/num_samples_ ) + log_term_
        eps1_ = d_hat + math.sqrt( 2*log_d_*(v_ + 2*d_hat)/num_samples_ ) + log_d_/(3*num_samples_)
        eps2_ = d_hat + math.sqrt( log_d_/(2*num_samples_) )
        eps_ = min(eps1_ , eps2_)

        # union bound
        r_hat = math.sqrt( math.log(4*count_union_bound/args.d)/(2*num_samples_) )
        d_hat = r_hat + math.sqrt( log_term_*log_term_ + 2*r_hat*log_d_/num_samples_ ) + log_term_
        eps_union_bound = d_hat + math.sqrt( log_d_/(2*num_samples_) )
    else:
        log_d_ = math.log(4/args.d)
        v_ = (1-mu_)*min(mu_,max_subg_freq_est)
        d_hat = est_devs_ + math.sqrt( log_d_/(2*num_trials_*num_samples_) )
        d_hat_2 = est_devs_ + math.sqrt( 2*max_subg_freq_est*log_d_/(num_trials_*num_samples_) )
        d_hat = min(d_hat,d_hat_2)
        eps_ = d_hat + math.sqrt( 2*log_d_*v_/num_samples_ )

        # union bound
        eps_union_bound = math.sqrt( math.log(2*count_union_bound/args.d)/(2*num_samples_) )


    return eps_ , bound_pseudo , eps_union_bound

print("Building search space...")
if args.geneexp == 0:
    searchspace = ps.create_selectors(data, ignore=[target_col])
    print("  len(searchspace) before",len(searchspace))
    #print(searchspace)
    for attr_name in [x for x in data.select_dtypes(include=['number']).columns.values if x not in [target_col]]:
        new_selectors_ = ps.create_numeric_selectors_for_attribute(data, attr_name, 5, False, None)
        searchspace.extend(new_selectors_[2:-2])
    print("  len(searchspace) after",len(searchspace))
    #print(searchspace)
else:
    cutoff_vals = np.linspace(0., data.max(numeric_only=True).max(), num=6, endpoint=False)
    cutoff_vals = np.delete(cutoff_vals,0)
    print("geneexp:",cutoff_vals)
    searchspace = []
    attr_list = [x for x in data.select_dtypes(include=['number']).columns.values if x not in [target_col]]
    for attr_name in tqdm(attr_list):
        searchspace_this = []
        max_attr = data[attr_name].max()
        min_attr = data[attr_name].min()
        for cutoff_val in cutoff_vals:
            if cutoff_val > min_attr and cutoff_val < max_attr:
                searchspace_this.append(ps.IntervalSelector(attr_name, float("-inf"), cutoff_val))
                searchspace_this.append(ps.IntervalSelector(attr_name, cutoff_val , float("inf")))
        searchspace.extend(searchspace_this)
        #print(searchspace_this)
    print("  len(searchspace) geneexp",len(searchspace))

print("Estimating most freq subgroup...")
max_subg_freq_est = 0.
for selector_ in tqdm(searchspace):
    max_subg_freq_est = max(max_subg_freq_est , selector_.covers(data).sum()/num_samples)
    #print('instances with ', str(alex_selector), alex_selector.covers(df))
print("   max_subg_freq_est",max_subg_freq_est)

# run correction
def run_correction(min_qual_=0.):
    devs_list = []
    run_times_list = []
    avg_dev = 0.
    avg_time_corr = 0.
    if args.corr == 1:
        if args.ub == 0:
            print("Running correction...")
            res_pool = []
            for i in range(args.res):
                # generate resample
                data_res = data.copy()
                if args.wy == 0:
                    if use_fixed_mu == 1:
                        data_res[target_col] = bernoulli.rvs(0.5, size=num_samples)
                    else:
                        data_res[target_col] = bernoulli.rvs(mu_hat, size=num_samples)
                else:
                    target_ = data_res[target_col].values.copy()
                    np.random.shuffle(target_)
                    data_res[target_col] = target_
                    #print("shuffled target",target_)
                    #print(data_res)
                #print(data_res)
                if use_fixed_mu == 1:
                    res_ = workers.apply_async(mine_sd , [data_res , 1 , target_col , max_depth , searchspace , 0.5 , min_qual_])
                else:
                    res_ = workers.apply_async(mine_sd , [data_res , 1 , target_col , max_depth , searchspace , mu_low , min_qual_])
                res_pool.append(res_)
            for res_ in tqdm(res_pool):
                res__ = res_.get()
                res_dev = res__[0]
                time_ = res__[1]
                if res_dev.shape[0] > 0:
                    res_dev_val = res_dev.loc[0]["quality"]
                else:
                    res_dev_val = min_qual_
                #print(res_dev_val)
                run_times_list.append(time_)
                devs_list.append(res_dev_val)
            print("Done correction!")
            #print("devs_list",devs_list)
            avg_dev = np.array(devs_list).mean()
        else:
            avg_dev = 0.
            run_times_list = [0.]
        eps , bound_pseudo , eps_union_bound = bound_FWER(avg_dev , mu_est , eps_mu , args.res , num_samples)
        devs_list.sort(reverse=True)

        run_times_list = np.array(run_times_list)
        avg_time_corr = run_times_list.mean()
        print("avg_time_corr",avg_time_corr)
        tot_time_corr = run_times_list.sum()
        print("  min_time",run_times_list.min())
        print("  max_time",run_times_list.max())
        print("  tot_time",tot_time_corr)
        #print(run_times_list)
        print("avg_dev",avg_dev)
        print("bound_pseudo",bound_pseudo)
        print("eps_union_bound",eps_union_bound)
        print("eps",eps)
        alpha_q = 0.
        if args.wy == 1:
            alpha_q = devs_list[ math.floor(args.d*len(devs_list)) ]
            print("alpha_q",alpha_q)
    return devs_list , avg_dev , eps , bound_pseudo , eps_union_bound , alpha_q , run_times_list , avg_time_corr , tot_time_corr


# run correction
devs_list , avg_dev , eps , bound_pseudo , eps_union_bound , alpha_q , run_times_list , avg_time_corr , tot_time_corr = run_correction()
tot_time_corr_full = tot_time_corr

if args.mine == 1:

    print("start mining")
    min_qual = min(eps , bound_pseudo)
    min_qual = eps
    if args.wy == 1:
        min_qual = alpha_q
    if args.ub == 1:
        min_qual = eps_union_bound
    res_ = mine_sd(data , args.k , target_col , max_depth , searchspace , mu_hat , min_qual)
    res_df = res_[0]
    time_mining = res_[1]
    print("done mining in",time_mining,"seconds")
    print(res_df[["quality","subgroup","relative_size_sg"]])

    if res_df.shape[0] > 0:
        print(res_df[["quality","subgroup","relative_size_sg"]].loc[0])

    sign_ = res_df["quality"] >= min_qual
    num_sign_results = sign_.sum()
    print("Number of significant results",num_sign_results)
    print("Fraction of significant results",float(num_sign_results)/float(args.k))
    if len(args.ores) > 0:
        res_df.to_csv(args.ores)


# write results to file
print_header = False
if os.path.isfile(args.o) == False:
    header = "db_name;num_samples;num_features;max_subg_length;cond;wy;union_bound;num_resamples;delta;signsd_fwer_bound;est_dev_q;bound_pseudo;alpha_q;eps_union_bound;avg_time_corr;tot_time_corr;tot_time_corr_full;time_mining;k;num_sign_results"
    print_header = True
fout = open(args.o,"a")
if print_header == True:
    fout.write(header+"\n")
out_res = args.db+";"+str(num_samples)+";"+str(num_features)+";"+str(args.maxd)+";"+str(args.cond)+";"+str(args.wy)+";"+str(args.ub)+";"+str(args.res)+";"+str(args.d)+";"+str(eps)+";"+str(avg_dev)+";"+str(bound_pseudo)+";"+str(alpha_q)+";"+str(eps_union_bound)+";"+str(avg_time_corr)+";"+str(tot_time_corr)+";"+str(tot_time_corr_full)+";"+str(time_mining)+";"+str(args.k)+";"+str(num_sign_results)
fout.write(out_res+"\n")
fout.close()
