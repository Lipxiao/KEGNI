import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='Argument Parser')
    main_group = parser.add_argument_group('main_group')
    kge_group = parser.add_argument_group('kge_group')
    mae_group = parser.add_argument_group('mae_group')

    main_group.add_argument("--max_steps", type=int,  default=100)
    main_group.add_argument('--device', type=int, default=0)

    main_group.add_argument('--dir', default='/media/disk/project/GRN/KEGNI/', type=str, help='prefix of dataset ')
    main_group.add_argument('--input', '-i',default='/media/disk/project/GRN/KEGNI/data/BEELINE/mESC_exp.csv', type=str, help='prefix of dataset ')
    main_group.add_argument('--eval', '-e', action="store_true", default=False, help='evaluate,only for dataset with ground truth')
    main_group.add_argument('--save_checkpoint', action="store_true", default=False, help='evaluate,only for dataset with ground truth')
    main_group.add_argument('--load_checkpoint', type=str, help='load checkpoint for training')
    main_group.add_argument('--norm', '-p', type=int, default='2', help='norm for pred construction')
    main_group.add_argument('--genes',type=int, default=None,help='norm for pred construction')
 

    mae_group.add_argument("--num_hidden", type=int, default=256, help="number of hidden units")
    mae_group.add_argument('--n_neighbors', '-n', type=int, default=30, help='n_neighbors for knn')
    mae_group.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads")
    mae_group.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    mae_group.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    mae_group.add_argument("--residual", action="store_true", default=False, help="use residual connection")
    mae_group.add_argument("--in_drop", type=float, default=.2, help="input feature dropout")
    mae_group.add_argument("--attn_drop", type=float, default=.1, help="attention dropout")
    # mae_group.add_argument("--norm", type=str, default=None)
    mae_group.add_argument("--mae_lr", type=float, default=1e-3,  help="learning rate")
    mae_group.add_argument("--mae_weight_decay", type=float, default=5e-4, help="weight decay")
    mae_group.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu for GAT")
    mae_group.add_argument("--activation", type=str, default="prelu")
    mae_group.add_argument("--mask_rate", type=float, default=0.3)
    mae_group.add_argument("--drop_edge_rate", type=float, default=0.0)
    mae_group.add_argument("--replace_rate", type=float, default=0.0)
    mae_group.add_argument("--encoder", type=str, default="gat")
    mae_group.add_argument("--decoder", type=str, default="gat")
    mae_group.add_argument("--loss_fn", type=str, default="sce")
    mae_group.add_argument("--alpha_l", type=float, default=1, help="`pow`coefficient for `sce` loss")
    mae_group.add_argument("--concat_hidden", action="store_true", default=False)
    
    kge_group.add_argument('--data_path',default='./data/KG/KEGG_mESC.tsv', type=str)
    kge_group.add_argument('--model', default='ComplEx', type=str)
    kge_group.add_argument('--negative_sample_size', default=16, type=int)
    kge_group.add_argument('-g', '--gamma', default=12.0, type=float)
    kge_group.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    kge_group.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    kge_group.add_argument('-b', '--batch_size', default=256, type=int)
    kge_group.add_argument('-r', '--regularization', default=0.00001, type=float)
    kge_group.add_argument('--kge_lr',  default=1e-3, type=float)

    args = parser.parse_args()

    return args
