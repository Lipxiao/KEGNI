n_neighbors=30
num_hidden=512
num_heads=4
num_layers=4

input=$1
genes=$2
norm=$3
max_steps=$4
data_path=$5
py_file=$6
lambda_kge=$7

python $py_file --eval \
    --n_neighbors $n_neighbors \
    --max_steps $max_steps \
    --data_path $data_path  \
    --input $input \
    --num_hidden $num_hidden \
    --num_heads $num_heads \
    --num_layers $num_layers \
    --norm $norm \
    --genes $genes \
    --lambda_kge $lambda_kge