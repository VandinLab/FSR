import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load subgroup results
df_res = pd.read_csv("subgroups-mnist-1.csv")
print(df_res)
sign_fig = np.zeros((26*26))
sign_set = set()
for subgroup in df_res["subgroup"]:
    #print(subgroup)
    feat_id = -1
    thr_val = 0
    flag = 0
    if "<" in subgroup:
        elems = subgroup.split("<")
        feat_id = int(elems[0])
        thr_str = elems[1]
        if "=" in thr_str:
            thr_str = thr_str[1:]
        thr_val = float(thr_str)
        flag = thr_val-1
    if ">" in subgroup:
        elems = subgroup.split(">")
        feat_id = int(elems[0])
        thr_str = elems[1]
        if "=" in thr_str:
            thr_str = thr_str[1:]
        thr_val = float(thr_str)
        flag = 1+thr_val
    if feat_id > 0:
        if feat_id not in sign_set:
            sign_set.add(feat_id)
            sign_fig[feat_id-1] = flag
            print("feature",feat_id,"at thr",thr_val,"flag",flag)

min_val = sign_fig.min()
max_val = sign_fig.max()
for i in range(sign_fig.shape[0]):
    if sign_fig[i] > 0.:
        sign_fig[i] = sign_fig[i]/max_val
#sign_fig[0] = 1
sign_fig = sign_fig.reshape((26,26))
plt.imsave('sign_pixels.pdf', sign_fig, cmap="bwr")
