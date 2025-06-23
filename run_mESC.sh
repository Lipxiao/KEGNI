data_path='data/KG/KEGG_mESC.tsv'
max_steps=1000

input='data/BEELINE/TF500/mESC_exp.csv'
genes=500

lambda_kge=1
norm=-1 #dot product
py_file='train.py'
sh  run_mESC_30_512_4_4.sh $input $genes $norm $max_steps $data_path $py_file $lambda_kge