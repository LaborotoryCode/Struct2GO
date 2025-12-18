import torch
import torch.nn.functional as F
import argparse
import numpy as np
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import pickle
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr,calculate_performance
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import warnings
import datetime
import dgl
import pandas as pd
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import requests



warnings.filterwarnings('ignore')
Thresholds = list(map(lambda x:round(x*0.01,2), list(range(1,100))))

if __name__ == "__main__":
    #参数设置
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-test_data', '--test_data',type=str,default='bp_test_plddt.pkl')
    parser.add_argument('-branch', '--branch',type=str,default='bp')
    parser.add_argument('-model','--model',type=str,default='save_models/mymodel_bp_8_0.0005_0.45.pkl')
    parser.add_argument('-labels_num', '--labels_num',type=int,default=809)
    parser.add_argument('-label_network', '--label_network', type=str, default='bp_label_network.dgl')
    args = parser.parse_args()
    labels_num = args.labels_num

    class DGLSafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'DGLHeteroGraph':
                return dgl.DGLGraph  # for older DGL versions
            return super().find_class(module, name) 
        
    def get_alphafold_plddt(uniprot_id): #TRY CHANGING DEFAULT
        parser = PDBParser(QUIET=True)

        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(url)

        plddts = []

        for line in response.text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                plddts.append(float(line[60:66]))
        
        #residue_plddt = [np.mean(plddts[i:i+4]) for i in range(0, len(plddts), 4)] #Aggregate pooling of plddts

        return plddts

    with open(args.test_data,'rb')as f:
        test_dataset = DGLSafeUnpickler(f).load()
    with open(args.label_network,'rb')as f:
        label_network, _ = dgl.load_graphs(args.label_network)
        label_network = label_network[0]
        print(type(label_network))
    model = torch.load(args.model)

    print("test", len(test_dataset))
    
    test_dataloader = GraphDataLoader(dataset=test_dataset, batch_size = 8,drop_last = False, shuffle = True)
    time = datetime.datetime.now()
    print(time)
    print('#########'+args.branch+'###########')
    print('########start testing###########') 


    t_loss = 0
    test_batch_num = 0
    pred = []
    actual = []
    model.eval()   
    #criterion = torch.nn.BCEWithLogitsLoss()
    for batched_graph, labels,sequence_feature  in test_dataloader:
            logits = model(batched_graph.to('cuda'), sequence_feature.to('cuda'),label_network.to('cuda'))
            #labels = torch.reshape(labels,(-1,labels_num))
            #labels = labels.reshape(-1, labels_num).float().to('cuda')
            labels = labels.reshape(labels.size(0), labels_num).float().to('cuda')
            print(labels.dim())
            loss = F.cross_entropy(logits,labels.to('cuda'))
            t_loss += loss.item()
            test_batch_num += 1
            pred.append(torch.sigmoid(logits).detach().cpu().numpy())  # (B, L)
            actual.append(labels.detach().cpu().numpy())      
    actual = np.concatenate(actual, axis=0)   # (N, L)
    pred = np.concatenate(pred, axis=0) 
            #writer.add_pr_curve('pr_curve',labels,logits,0)
    test_loss = "{}".format(t_loss / test_batch_num)    
    #writer.add_scalar('test/loss',test_loss,epoch)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)

    # average number of labels per protein
    print("Avg labels per protein:", actual.sum(axis=1).mean())

    # fraction of positives
    print("Positive rate:", actual.mean())

    # proteins with no labels
    print("Zero-label proteins:", np.sum(actual.sum(axis=1) == 0))
    
    auc_values = []
    print("here", actual)
    n_labels = actual.shape[1]
    print(n_labels)
    for j in range(n_labels):
        print(j)
        fpr, tpr, _ = roc_curve(actual.flatten(), pred.flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        auc_values.append(auc_score)

    aupr=cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
    aupr_values = []
    y_true = np.array(actual) 
    y_scores = np.array(pred)
    n_labels = y_true.shape[1]
    for k in range(n_labels):
        aupr1 = average_precision_score(y_true[:, k], y_scores[:, k])
        aupr_values.append((1-0.3)*aupr1 + 0.3*aupr)

    score_dict = {}
    each_best_fcore = 0
    #best_fscore = 0
    each_best_scores = []
    #writer.add_pr_curve('pr_curve',actual,pred,0,num_thresholds=labels_num)
    for i in range(len(Thresholds)):
        print("threshy", i)
        f_score,precision, recall  = calculate_performance(actual, pred, label_network,threshold=Thresholds[i])
        if f_score >= each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score,auc_values,aupr_values]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores        
    t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    precision, auc_score = each_best_scores[3], each_best_scores[4] 
    # auc_values, aupr_values = each_best_scores[5],each_best_scores[6]
    print('testloss:{},t:{},f_score{}, auc{}, recall{}, precision{},aupr{}'.format(
        test_loss, t, f_score, auc_score, recall, precision,aupr))  
        
    

    # print('f_score: {}'.format(f_score))
    # print('auc_values: {}'.format(auc_values))
    # print('aupr_values: {}'.format(aupr_values))   
    # df1 = pd.DataFrame(f_score)
    # df2 = pd.DataFrame(auc_values)
    # df3 = pd.DataFrame(aupr_values)
    # df1.to_excel('f_score.xlsx', index=False, engine='openpyxl')
    # df2.to_excel('auc_values.xlsx', index=False, engine='openpyxl')
    # df3.to_excel('aupr_values.xlsx', index=False, engine='openpyxl')
    
    # bins = [i/10 for i in range(11)]
    # # 设置柱状图的宽度和位置
    # width = (bins[1] - bins[0]) / 4  # 使得三个柱子在一个bin内紧密相邻，但是不同bin之间有空隙



    # # 手动计算每组数据的直方图
    # hist_data1, _ = np.histogram(f_score, bins=bins)
    # hist_data2, _ = np.histogram(auc_values, bins=bins)
    # hist_data3, _ = np.histogram(aupr_values, bins=bins)

    # # 为每组数据设置中心点位置
    # centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    # centers1 = [center - width for center in centers]
    # centers2 = centers
    # centers3 = [center + width for center in centers]


    # # 绘制三组数据的柱状图
    # plt.bar(centers1, hist_data1, width=width, alpha=0.5, label='f_score', edgecolor='black')
    # plt.bar(centers2, hist_data2, width=width, alpha=0.5, label='auc_values', edgecolor='black')
    # plt.bar(centers3, hist_data3, width=width, alpha=0.5, label='aupr_values', edgecolor='black')

    # plt.title('Distribution of BP Test Data Sets')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.xticks(bins)
    # plt.legend(loc='upper left')  # 显示图例

    # plt.savefig('histogram3.svg', format='svg')

    # plt.show()
                
