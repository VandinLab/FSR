# Efficient Discovery of Significant Patterns with Few-Shot Resampling

This readme describes the code for the FSR algorithm to mine significant patterns with few-shots resampling.


## Running FSR

The FSR algorithm to mine significant subgroups can be executed with the script `fsr-alg.py`. It accepts several parameters that are listed and described with the `-h` option (see below for more details):
```
usage: fsr-alg.py [-h] [-db DB] [-target TARGET] [-tval TVAL] [-k K]
                  [-maxd MAXD] [-eps EPS] [-corr CORR] [-mine MINE] [-res RES]
                  [-p P] [-pn PN] [-dfs DFS] [-simp SIMP] [-wy WY] [-ub UB]
                  [-cond COND] [-o O] [-ores ORES] [-d D] [-cat CAT]
                  [-geneexp GENEEXP] [-head HEAD]

optional arguments:
  -h, --help        show this help message and exit
  -db DB            path to input file
  -target TARGET    string of target column
  -tval TVAL        value of target to consider
  -k K              number of top-k results to find (def.=10000)
  -maxd MAXD        max number of conjunction terms (def.=3)
  -eps EPS          default value of eps to use for output (ignored when
                    corr=1)
  -corr CORR        run correction (def.=1)
  -mine MINE        run mining (def.=1)
  -res RES          number of resamples (def.=10)
  -p P              parallel computations (def.=1)
  -pn PN            number of parallel cores (def.=0 use all)
  -dfs DFS          use dfs exploration (def.=1)
  -simp SIMP        use simple exploration (def.=0)
  -wy WY            run WY correction (def.=0)
  -ub UB            use union bound (def.=0)
  -cond COND        use conditional distribution correction (def.=0)
  -o O              output path
  -ores ORES        output path for significant subgroups (def. no output)
  -d D              delta (def. 0.05)
  -cat CAT          categorize data (def. 0)
  -geneexp GENEEXP  custom search space for gene expression data (def. 0)
  -head HEAD        1 if data has an header (def. 0)
```

## Input format

The `fsr-alg.py` script accepts an input dataset as a comma separated value file. The path for the input file must be given with the `-db` argument.
It is then necessary to provide the column name (or number) of the target feature (argument `-target`), and the target value to consider as value `1` (argument `-tval`), while all other values are set to target `0`. The script assumes that the input dataset does not have an header to parse the column name. In case an header is included in the dataset, use the argument `-head 1`.

## FSR parameters

The `fsr-alg.py` script mines the `k` most significant subgroups, where `k` is specified with the argument `-k` (default set to `10000`).
The maximum number of conditions to include in any subgroup is specified with the argument `-maxd` (default set to `3`).
The number of resamples of the target labels is set with the argument `-res` (default set to `10`).
The script outputs the experiment statistics (running time, number of significant patterns, ...) in a comma separated .csv file with the path given to the argument `-o` (default set to `results_signfsr.csv`).
The significant patterns can be saved to a .csv file using the argument `-ores`.

## Reproduce the experiments described in the paper

1. Download the `datasets.zip` archive from this link: https://tinyurl.com/FSRdatasets
2. move and unzip the archive in the `/data/` folder
3. install the required python 3 packages with the command `pip install -r requirements.txt`
4. run all experiments of "Impact of parameters on FSR" paragraph with the command `python run_experiments_params.py`
5. run all experiments of "Evaluation of FSR-C" and "Evaluation of FSR-U" paragraphs with the command `python run_all_experiments.py`
6. run all experiments of "Application to Neural Network interpretation" paragraph with the command `python run_mnist.py`

## Contacts

You can contact us at leonardo.pellegrina@unipd.it and fabio.vandin@unipd.it for any questions and for reporting bugs.
