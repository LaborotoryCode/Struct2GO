from cProfile import label
from random import shuffle
from re import T
from statistics import mode
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
#from dgl.heterograph import DGLHeteroGraph
from sklearn.preprocessing import MultiLabelBinarizer
import dgl
from model.network import SAGNetworkHierarchical,SAGNetworkGlobal
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from tkinter import _flatten
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import argparse
import warnings
import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib   
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr,calculate_performance
import io
import requests
from Bio.PDB import PDBParser
from collections import OrderedDict


warnings.filterwarnings('ignore')
Thresholds = list(map(lambda x:round(x*0.01,2), list(range(1,100))))

# def cacul_aupr(lables, pred):
#     precision, recall, _thresholds = metrics.precision_recall_curve(lables, pred)
#     aupr = metrics.auc(recall, precision)
#     return aupr

# def calculate_performance(actual, pred_prob, threshold=0.2, average='micro'):
#     pred_lable = []
#     actual_label = []
#     for l in range(len(pred_prob)):
#         eachline = (np.array(pred_prob[l]) > threshold).astype(np.int32)
#         #eachline = eachline.tolist()
#         #pred_lable.append(list(_flatten(eachline)))
#         pred_lable.append(list(eachline))
#     for l in range(len(actual)):
#         eachline = (np.array(actual[l])).astype(np.int32)
#         #eachline = eachline.tolist()
#         #actual_label.append(list(_flatten(eachline)))
#         actual_label.append(list(eachline))
#     f_score = f1_score(actual_label, pred_lable, average=average)
#     recall = recall_score(actual_label, pred_lable, average=average)
#     precision = precision_score(actual_label,  pred_lable, average=average)
#     return f_score, precision, recall    


if __name__ == "__main__":
    #参数设置 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=128,help="the number of the bach size") #hmmm
    parser.add_argument('-learningrate', '--learningrate',type=float,default=5e-4)
    parser.add_argument('-dropout', '--dropout',type=float,default=0.45)
    parser.add_argument('-train_data', '--train_data',type=str,default="train_plddt.pkl")
    parser.add_argument('-valid_data', '--valid_data',type=str,default="valid_plddt.pkl")
    parser.add_argument('-branch', '--branch',type=str,default='mf')
    parser.add_argument('-labels_num', '--labels_num',type=int,default=273)
    parser.add_argument('-label_network', '--label_network', type=str, default="label_network.dgl")

    args = parser.parse_args()

    e = 0

    #Adding plddt structures for confidence awareness in predictions
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


    class DGLSafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'DGLHeteroGraph':
                return dgl.DGLGraph  # for older DGL versions
            return super().find_class(module, name) 

    with open(args.train_data,'rb')as f:
        train_dataset = DGLSafeUnpickler(f).load()

    with open(args.valid_data,'rb')as f:
        valid_dataset = DGLSafeUnpickler(f).load()
    
        #dgl.save_graphs("valid_dataset.dgl", [valid_dataset])

    with open(args.label_network,'rb')as f:
        label_network, _ = dgl.load_graphs(args.label_network)
        label_network = label_network[0]
        print(type(label_network))


    # class MyDataSet(Dataset):
    #     def __init__(self,emb_graph,emb_seq_feature,emb_label):
    #         super().__init__()
    #         self.list = list(emb_graph.keys())
    #         self.graphs = emb_graph
    #         self.seq_feature = emb_seq_feature
    #         self.label = emb_label

    #     def __getitem__(self,index): 
    #         protein = self.list[index] 
    #         graph = self.graphs[protein]
    #         seq_feature = self.seq_feature[protein]
    #         label = self.label[protein]

    #         return graph, label, seq_feature 

    #     def __len__(self):
    #         return  len(self.list) 

    batch_size = args.batch_size
    learningrate = args.learningrate
    dropout = args.dropout

    # dataset = MyDataSet(emb_graph = emb_graph,emb_seq_feature = emb_seq_feature,emb_label = emb_label)
    #train_size = int(len(train_dataset) * 0.8)
    #print(train_size)
    # test_size = len(dataset) - train_size
    # #trash_size = len(dataset) - train_size - test_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
 
    dataloader = GraphDataLoader(dataset=train_dataset, batch_size = batch_size,drop_last = False, shuffle = True)
    valid_dataloader = GraphDataLoader(dataset=valid_dataset, batch_size = 1,drop_last = False, shuffle = True)
    time = datetime.datetime.now()
    print(time)
    print('#########'+args.branch+'###########')
    print('########start training###########')
    labels_num = args.labels_num

    #Commented out
    num_convs_nums = [1,2,3,4]
    plt.figure("P-R Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    print("num_convs")           
    for num_convs in num_convs_nums:
        print(num_convs) 
    #
    
    model = SAGNetworkHierarchical(56,512,labels_num,num_convs=2,pool_ratio=0.75,dropout=dropout).to('cuda')
    #model = SAGNetworkGlobal(56,512,labels_num,dropout=dropout).to('cuda')
    #optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=learningrate, weight_decay=0.001)
    loss_fcn = nn.CrossEntropyLoss()
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    best_fscore = 0
    best_scores = []
    best_score_dict = {}

    loss_history = []
    i=1

    for epoch in range(10): #Always gives you one less epoch for some reason

        i+=1

        print("started epoch")
        model.train()
        _loss = 0
        batch_num = 0
        train_pred = []
        train_actual = []

        for batched_graph, labels,sequence_feature in dataloader:
            print("In the for batched loop")
            logits = model(batched_graph.to('cuda'),sequence_feature.to('cuda'),label_network.to('cuda'))
            labels = torch.reshape(labels,(-1,labels_num))

            print(f"DEBUG: Shape of model output (logits): {logits.shape}")
            print(f"DEBUG: Shape of labels tensor: {labels.shape}")
            print(f"DEBUG: Number of individual graphs in DGL batch: {batched_graph.batch_size}") # Important!
            print(f"DEBUG: Total number of nodes in DGL batch: {batched_graph.num_nodes()}")

            loss = F.cross_entropy(logits,labels.to('cuda'))
            # F.binary_cross_entropy()
            #loss = F.binary_cross_entropy(logits,labels.to('cuda'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss += loss.item()
            batch_num += 1
            train_pred.append(torch.sigmoid(logits).tolist())
            train_actual.append(labels.tolist())
            i+=1
        print(i)
        epoch_loss = "Epoch Loss: {}".format(_loss / batch_num)
        loss_history.append(epoch_loss)
        print(epoch_loss)


        # fpr, tpr, th = roc_curve(np.array(train_actual).flatten(), np.array(train_pred).flatten(), pos_label=1)
        # auc_score = auc(fpr, tpr)
        # aupr=cacul_aupr(np.array(train_actual).flatten(), np.array(train_pred).flatten())
        # score_dict = {}
        # each_best_fcore = 0
        # #best_fscore = 0
        # each_best_scores = []
        # #writer.add_pr_curve('pr_curve',actual,pred,0,num_thresholds=labels_num)
        # for i in range(len(Thresholds)):
        #     f_score,precision, recall  = calculate_performance(train_actual, train_pred, label_network,threshold=Thresholds[i])
        #     if f_score >= each_best_fcore:
        #         each_best_fcore = f_score
        #         each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
        #         scores = [f_score, recall, precision, auc_score]
        #         score_dict[Thresholds[i]] = scores
        # if each_best_fcore >= best_fscore:
        #     best_fscore = each_best_fcore
        #     best_scores = each_best_scores
        #     best_score_dict = score_dict
        #     torch.save(model, '/home/jiaops/lyjps/save_models/mymodel_{}_{}_{}_{}.pkl'.format(args.branch,batch_size,learningrate,dropout))
        # t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        # precision, auc_score = each_best_scores[3], each_best_scores[4] 
        # print('########training metric###########')
        # print('epoch{},loss{},testloss:{},t:{},f_score{}, auc{}, recall{}, precision{},aupr{}'.format(
        #         epoch, epoch_loss, epoch_loss, t, f_score, auc_score, recall, precision,aupr))


        t_loss = 0
        valid_batch_num = 0
        pred = []
        actual = []
        model.eval()
        print("This place?")

        all_preds_tensors = []
        all_actuals_tensors = []

        with torch.no_grad():
            """
            for batched_graph, labels,sequence_feature  in dataloader:
                print("Running validation")
                logits = model(batched_graph.to('cuda'),sequence_feature.to('cuda'),label_network.to('cuda'))
                labels = torch.reshape(labels,(-1,labels_num))
                loss = F.cross_entropy(logits,labels.to('cuda'))
                #loss = F.binary_cross_entropy(logits,labels.to('cuda'))
                t_loss += loss.item()
                valid_batch_num += 1
                all_preds_tensors.append(torch.sigmoid(logits).cpu())
                all_actuals_tensors.append(labels.cpu())
                #writer.add_pr_curve('pr_curve',labels,logits,0)

            final_preds_tensor = torch.cat(all_preds_tensors, dim=0)
            final_actuals_tensor = torch.cat(all_actuals_tensors, dim=0)
            actual = final_actuals_tensor.numpy()
            pred = final_preds_tensor.numpy()

            print("pred shape og:", type(pred), len(pred), pred[:5])
            print("actual shape:", actual.shape)
            print("unique values in actual:", np.unique(actual))

            test_loss = "{}".format(t_loss / valid_batch_num)    
            #writer.add_scalar('test/loss',test_loss,epoch)
            fpr, tpr, th = roc_curve(actual, pred, pos_label=1)
            auc_score = auc(fpr, tpr)
            """
            for batched_graph, labels,sequence_feature  in dataloader:
                logits = model(batched_graph.to('cuda'),sequence_feature.to('cuda'),label_network.to('cuda'))
                labels = torch.reshape(labels,(-1,labels_num))
                loss = F.cross_entropy(logits,labels.to('cuda'))
                #loss = F.binary_cross_entropy(logits,labels.to('cuda'))
                t_loss += loss.item()
                valid_batch_num += 1
                pred.append(torch.sigmoid(logits).tolist())
                actual.append(labels.tolist())
                #writer.add_pr_curve('pr_curve',labels,logits,0)
            test_loss = "{}".format(t_loss / valid_batch_num)    
            #mlb = MultiLabelBinarizer()
            #print(actual[0])
            #actual_binary = mlb.fit_transform(actual)
            #writer.add_scalar('test/loss',test_loss,epoch)
            fpr, tpr, th = roc_curve(np.vstack(actual).flatten(), np.vstack(pred).flatten(), pos_label=1)
            auc_score = auc(fpr, tpr)
            aupr=cacul_aupr(np.vstack(actual).flatten(), np.vstack(pred).flatten())
            score_dict = {}
            each_best_fcore = 0
            #best_fscore = 0
            each_best_scores = []
            #writer.add_pr_curve('pr_curve',actual,pred,0,num_thresholds=labels_num)

            for i in range(0, len(Thresholds),3):
                f_score,precision, recall  = calculate_performance(np.vstack(actual), np.vstack(pred), label_network,threshold=Thresholds[i])
                if f_score >= each_best_fcore:
                    each_best_fcore = f_score
                    each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
                    scores = [f_score, recall, precision, auc_score]
                    score_dict[Thresholds[i]] = scores
            if each_best_fcore >= best_fscore:
                best_fscore = each_best_fcore
                best_scores = each_best_scores
                best_score_dict = score_dict
                #torch.save(model, "C:/Users/ayaan/Downloads/Struct2GO-master/Struct2GO-master/save_models/_{}_{}_{}_{}.pkl".format(args.branch,batch_size,learningrate,dropout))
            t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
            precision, auc_score = each_best_scores[3], each_best_scores[4] 
            print('########valid metric###########')
            print('epoch{},loss{},testloss:{},t:{},f_score={}, auc={}, recall={}, precision={},aupr={}'.format(
                    epoch, epoch_loss, test_loss, t, f_score, auc_score, recall, precision,aupr))
        #precision, recall, thresholds = precision_recall_curve(np.array(actual).flatten(), np.array(pred).flatten())
            plt.plot(recall,precision,label = "num_convs="+str(num_convs))

    #plt.legend()
    #plt.savefig('/home/jiaops/lyjps/processed_data/pr_num_convs.jpg')       

        #fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        #auc_score = auc(fpr, tpr)
        #f_score,precision, recall  = calculate_performance(actual, pred)
        #print('f_score{},precision{},recall{}'.format(f_score,precision, recall))

    plt.plot(loss_history, label="Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss over Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_loss_plot.png")
    plt.show()
