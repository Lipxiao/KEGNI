
n_neighbors=30
num_hidden=256
num_heads=4
num_layers=2

input=$1
genes=$2
norm=$3
data_path=$4
max_steps=$5
python train.py \
    --n_neighbors $n_neighbors \
    --max_steps $max_steps \
    --data_path $data_path  \
    --input $input \
    --num_hidden $num_hidden \
    --num_heads $num_heads \
    --num_layers $num_layers \
    --norm $norm \
    --genes $genes