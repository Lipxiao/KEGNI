
input='data/PBMC/naiveCD4Tcells_exp.csv'
genes=100 # No significance for non-BEELINE datasets.
norm=-1 
data_path='data/KG/KEGG_NaiveCD4+Tcell.tsv'
max_steps=1000
sh run_pbmc_30_256_4_2.sh $sh $input $genes $norm $data_path $max_steps