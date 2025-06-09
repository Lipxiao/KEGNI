from utils import EarlyPrec,computeScores
import pandas as pd
import argparse
def parser_args():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument("--predDF", "-p",type = str,help='predicted edges dataframe',
                        default="/media/disk/project/crosstalk/BEELINE/Beeline/outputs/scRNA/hESC/PPCOR/rankedEdges.csv")
    parser.add_argument('--trueDF', '-t',type = str, help='True edges dataframe',
                        default="/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/hESC/hESC-ChIP-network.csv")
    args = parser.parse_args()
    return args

def MultiEval(predDF,trueDF):
    predDF.columns = ['Gene1','Gene2','EdgeWeight']

    predDF['Gene1'] = predDF['Gene1'].str.upper()
    predDF['Gene2'] = predDF['Gene2'].str.upper()
    unique_genes = pd.concat([predDF['Gene2'], predDF['Gene1']]).unique()

    netDF = trueDF.iloc[:, :2].copy()
    netDF.columns = ['Gene1','Gene2']
    netDF['Gene1'] = netDF['Gene1'].str.upper()
    netDF['Gene2'] = netDF['Gene2'].str.upper()
    netDF = netDF[(netDF.Gene1.isin(unique_genes)) & (netDF.Gene2.isin(unique_genes))]
    # Remove self-loops.
    netDF = netDF[netDF.Gene1 != netDF.Gene2]
    # Remove duplicates (there are some repeated lines in the ground-truth networks!!!). 
    netDF.drop_duplicates(keep = 'first', inplace=True)
    trueEdgesDF = netDF.copy()
    unique_gene1 = netDF['Gene1'].unique()
    unique_gene2_combined = pd.concat([netDF['Gene2'], netDF['Gene1']]).unique()
    predEdgeDF = predDF[
        predDF['Gene1'].isin(unique_gene1) &
        predDF['Gene2'].isin(unique_gene2_combined)
    ].copy()

    epr = EarlyPrec(trueEdgesDF,predEdgeDF)
    prec, recall, fpr, tpr,pr,roc = computeScores(trueEdgesDF,predEdgeDF)
    return epr,pr,roc

if __name__ == '__main__':
    args = parser_args()
    predDF = pd.read_csv(args.predDF, sep = ',',header = 0, index_col = None)
    trueDF = pd.read_csv(args.trueDF, sep = ',',header = 0, index_col = None)
    epr,pr,roc = MultiEval(predDF,trueDF)
    print(f"Early Precision: {epr}")
    print(f"Precision: {pr}")
    print(f"ROC: {roc}")
    
