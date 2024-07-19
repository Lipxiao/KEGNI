from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, average_precision_score
import torch
from itertools import product, permutations, combinations, combinations_with_replacement
import pandas as pd
import numpy as np
import os
from itertools import product, permutations
import random

def computeScores(trueEdgesDF, predEdgeDF,
                  directed=True, selfEdges=True):
    '''        
    Computes precision-recall and ROC curves
    using scikit-learn for a given set of predictions in the 
    form of a DataFrame.
    :param trueEdgesDF:   A pandas dataframe containing the true classes.The indices of this dataframe are all possible edgesin a graph formed using the genes in the given dataset. This dataframe only has one column to indicate the classlabel of an edge. If an edge is present in the reference network, it gets a class label of 1, else 0.
    :type trueEdgesDF: DataFrame
    :param predEdgeDF:   A pandas dataframe containing the edge ranks from the prediced network. The indices of this dataframe are all possible edges.This dataframe only has one column to indicate the edge weightsin the predicted network. Higher the weight, higher the edge confidence.
    :type predEdgeDF: DataFrame
    :param directed:   A flag to indicate whether to treat predictionsas directed edges (directed = True) or undirected edges (directed = False).
    :type directed: bool
    :param selfEdges:   A flag to indicate whether to includeself-edges (selfEdges = True) or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: bool
    :returns:
            - prec: A list of precision values (for PR plot)
            - recall: A list of precision values (for PR plot)
            - fpr: A list of false positive rates (for ROC plot)
            - tpr: A list of true positive rates (for ROC plot)
            - AUPRC: Area under the precision-recall curve
            - AUROC: Area under the ROC curve
    '''

    if directed:
        # Initialize dictionaries with all
        # possible edges
        if selfEdges:
            possibleEdges = list(product(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]),
                                         repeat=2))
        else:
            possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]),
                                              r=2))

        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth
        value = [0] * len(possibleEdges)

        trueEdges = list(trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2'])
        data = {'possibleEdges': list({'|'.join(p) for p in possibleEdges}), 'value': value}
        possibleEdgesDF = pd.DataFrame(data)
        possibleEdgesDF.loc[(possibleEdgesDF.loc[:, 'possibleEdges'].isin(trueEdges)), 'value'] = 1
        TrueEdgeDict = dict(zip(possibleEdgesDF['possibleEdges'], possibleEdgesDF['value']))

        predEdgeDF.loc[:, 'Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
        predEdgeDF.index = predEdgeDF.loc[:, 'Edges']
        # data = {'possibleEdges': list({'|'.join(p) for p in possibleEdges}), 'value': value}
        possibleEdgesDF = pd.DataFrame(data)
        filter_edges = possibleEdgesDF['possibleEdges'].isin(predEdgeDF.loc[:, 'Edges'])
        possibleEdgesDF.loc[filter_edges, 'value'] = predEdgeDF.loc[possibleEdgesDF[filter_edges]['possibleEdges'], 'EdgeWeight'].values
        PredEdgeDict = dict(zip(possibleEdgesDF['possibleEdges'], possibleEdgesDF['value']))

    else:

        # Initialize dictionaries with all
        # possible edges
        if selfEdges:
            possibleEdges = list(combinations_with_replacement(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]),
                                                               r=2))
        else:
            possibleEdges = list(combinations(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]),
                                              r=2))
        TrueEdgeDict = {'|'.join(p): 0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p): 0 for p in possibleEdges}

        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth

        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[((trueEdgesDF['Gene1'] == key.split('|')[0]) &
                                    (trueEdgesDF['Gene2'] == key.split('|')[1])) |
                                   ((trueEdgesDF['Gene2'] == key.split('|')[0]) &
                                    (trueEdgesDF['Gene1'] == key.split('|')[1]))]) > 0:
                TrueEdgeDict[key] = 1

        # Compute PredEdgeDict Dictionary
        # from predEdgeDF

        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[((predEdgeDF['Gene1'] == key.split('|')[0]) &
                                    (predEdgeDF['Gene2'] == key.split('|')[1])) |
                                   ((predEdgeDF['Gene2'] == key.split('|')[0]) &
                                    (predEdgeDF['Gene1'] == key.split('|')[1]))]
            if len(subDF) > 0:
                PredEdgeDict[key] = max(np.abs(subDF.EdgeWeight.values))

    outDF = pd.DataFrame([TrueEdgeDict, PredEdgeDict]).T
    outDF.columns = ['TrueEdges', 'PredEdges']

    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)

    return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr)


def EarlyPrec(trueEdgesDF, predEdgeDF):

    # predEdgeDF['Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
    predEdgeDF.loc[:, 'Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
    # limit the predicted edges to the genes that are in the ground truth
    Eprec = {}
    # Consider only edges going out of TFs

    trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
    trueEdgesDF.drop_duplicates(keep='first', inplace=True)
    trueEdgesDF.reset_index(drop=True, inplace=True)

    uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
    possibleEdges_TF = set(product(set(trueEdgesDF.Gene1), set(uniqueNodes)))

    # Get a list of all possible interactions
    possibleEdges_noSelf = set(permutations(uniqueNodes, r=2))

    # Find intersection of above lists to ignore self edges
    possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)

    TrueEdgeDict = {'|'.join(p): 0 for p in possibleEdges}

    trueEdges = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
    trueEdges = trueEdges[trueEdges.isin(TrueEdgeDict)]
    numEdges = len(trueEdges)

    predDF_new = predEdgeDF[predEdgeDF.loc[:, 'Edges'].isin(TrueEdgeDict)]

    # Use num True edges or the number of
    # edges in the dataframe, which ever is lower
    maxk = min(predDF_new.shape[0], numEdges)
    edgeWeightTopk = predDF_new.iloc[maxk-1].EdgeWeight

    nonZeroMin = np.nanmin(predDF_new.EdgeWeight.replace(0, np.nan).values)
    bestVal = max(nonZeroMin, edgeWeightTopk)

    newDF = predDF_new.loc[(predDF_new['EdgeWeight'] >= bestVal)]
    rankDict = set(newDF['Gene1'] + "|" + newDF['Gene2'])

    # Erec = {}
    intersectionSet = rankDict.intersection(trueEdges)
    Eprec = len(intersectionSet)/len(rankDict)
    # Erec = len(intersectionSet)/len(trueEdges)

    return Eprec


def save_ckpt(step, model, optimizer, model_name, ckpt_folder):

    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    torch.save(
        {
            'step': step,
            # 'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': lr_scheduler.state_dict()
        },
        f'{ckpt_folder}{model_name}.pth'
    )

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
    