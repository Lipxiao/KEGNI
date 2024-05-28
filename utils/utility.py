from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, average_precision_score
import torch
from itertools import product, permutations, combinations, combinations_with_replacement
import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
sns.set(rc={"lines.linewidth": 2}, palette="deep", style="ticks")


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

        predEdgeDF['Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
        predEdgeDF.index = predEdgeDF['Edges']
        # data = {'possibleEdges': list({'|'.join(p) for p in possibleEdges}), 'value': value}
        possibleEdgesDF = pd.DataFrame(data)
        filter_edges = possibleEdgesDF['possibleEdges'].isin(predEdgeDF['Edges'])
        possibleEdgesDF.loc[filter_edges, 'value'] = predEdgeDF.loc[possibleEdgesDF[filter_edges]['possibleEdges'], 'EdgeWeight'].values
        PredEdgeDict = dict(zip(possibleEdgesDF['possibleEdges'], possibleEdgesDF['value']))

    # if directed:
    #     # Initialize dictionaries with all
    #     # possible edges
    #     if selfEdges:
    #         possibleEdges = list(product(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
    #                                      repeat = 2))
    #     else:
    #         possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
    #                                      r = 2))

    #     TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
    #     PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}

    #     # Compute TrueEdgeDict Dictionary
    #     # 1 if edge is present in the ground-truth
    #     # 0 if edge is not present in the ground-truth
    #     for key in TrueEdgeDict.keys():
    #         if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
    #                (trueEdgesDF['Gene2'] == key.split('|')[1])])>0:
    #                 TrueEdgeDict[key] = 1

    #     for key in PredEdgeDict.keys():
    #         subDF = predEdgeDF.loc[(predEdgeDF['Gene1'] == key.split('|')[0]) &
    #                            (predEdgeDF['Gene2'] == key.split('|')[1])]
    #         if len(subDF)>0:
    #             PredEdgeDict[key] = np.abs(subDF.EdgeWeight.values[0])

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

    # outDF = pd.DataFrame([TrueEdgeDict,PredEdgeDict]).T
    # outDF.columns = ['TrueEdges','PredEdges']
    # prroc = importr('PRROC')
    # prCurve = prroc.pr_curve(scores_class0 = FloatVector(list(outDF['PredEdges'].values)),
    #           weights_class0 = FloatVector(list(outDF['TrueEdges'].values)))

    # fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
    #                                  y_score=outDF['PredEdges'], pos_label=1)

    # prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
    #                                                   probas_pred=outDF['PredEdges'], pos_label=1)

    # return prec, recall, fpr, tpr, prCurve[2][0], auc(fpr, tpr)

    outDF = pd.DataFrame([TrueEdgeDict, PredEdgeDict]).T
    outDF.columns = ['TrueEdges', 'PredEdges']

    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)

    return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr)


def EarlyPrec(trueEdgesDF, predEdgeDF):

    predEdgeDF['Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
    # limit the predicted edges to the genes that are in the ground truth
    Eprec = {}
    from itertools import product, permutations
    import os
    # Consider only edges going out of TFs

    trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
    trueEdgesDF.drop_duplicates(keep='first', inplace=True)
    trueEdgesDF.reset_index(drop=True, inplace=True)

    uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
    possibleEdges_TF = set(product(set(trueEdgesDF.Gene1), set(uniqueNodes)))

    # Get a list of all possible interactions
    possibleEdges_noSelf = set(permutations(uniqueNodes, r=2))

    # Find intersection of above lists to ignore self edges
    # TODO: is there a better way of doing this?
    possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)

    TrueEdgeDict = {'|'.join(p): 0 for p in possibleEdges}

    trueEdges = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
    trueEdges = trueEdges[trueEdges.isin(TrueEdgeDict)]
    numEdges = len(trueEdges)

    predDF_new = predEdgeDF[predEdgeDF['Edges'].isin(TrueEdgeDict)]

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


def get_scores(edges_pos, edges_neg, adj_rec):

    # Predict on test set of edges
    preds = []
    # pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append((adj_rec[e[0], e[1]].item()))
        # pos.append(adj_orig[e[0], e[1]])

    # Predict on test set of negative edges
    preds_neg = []
    # neg = []
    for e in edges_neg:

        preds_neg.append((adj_rec[e[0], e[1]].data))
        # neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score, preds_all, labels_all


def get_acc(adj_label, adj_rec):
    data_adj = torch.zeros((adj_rec.shape[0], adj_rec.shape[0]), dtype=torch.float32)
    data_adj[adj_label['source_node_id'].values, adj_label['target_node_id'].values] = 1.0
    labels_all = data_adj.view(-1).long()
    # pred[pred < 0.7] = 0
    preds_all = (adj_rec > 0.8).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy
