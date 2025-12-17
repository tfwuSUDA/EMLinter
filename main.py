# %%

# coding: utf-8

# %%


class config():
    def __init__(self):
        self.vocab_size=65
        self.strvocab_size=3
        self.embedding_size=100
        self.strembedding_size=2
        self.feature_size=64
#         self.mi_kernel_size=[[1],[1,3],[1,3,5],[1,3,5,7]]
#         self.lnc_kernel_size=[[1],[1,3,5],[1,3,5,9],[1,3,5,9,15]]
#         self.mi_kernel_size=[2,4,6]
#         self.lnc_kernel_size=[4,8,16]
        self.mi_kernel_size=[1,3,5]
        self.lnc_kernel_size=[3,7,11]
        self.max_mi=25
        self.max_lnc=22743
        self.num_class=2
        


# %%


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




from models import *




import numpy as np
import time
import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import *
from sklearn.model_selection import KFold,StratifiedKFold
from tqdm import tqdm 
from torch.utils.data import DataLoader,Dataset,ConcatDataset


# %%


parser = argparse.ArgumentParser(description="type args")
parser.add_argument('-batch_size', type=int, default=32, help="BatchSize")
parser.add_argument('-epochs', type=int, default=91, help="Epoch Num")
parser.add_argument('-train', type=str, default='True', help="Train or Eval")
parser.add_argument('-cuda_id', type=int, default='0', help="CUDA Num")
parser.add_argument('-model_save_path', type=str,
                    default='./model', help="model save path")

args = parser.parse_args(args=[])


class Data(Dataset):
    def __init__(self,txtpath):
        if args.cuda_id != -1:
            device = 'cuda:{}'.format(args.cuda_id)
        else:
            device = 'cpu'

        self.data = txtpath
        
        self.featurelnc=pd.read_csv('../extracted_feature/lncfeature.csv',index_col=0)
        self.featuremi=pd.read_csv('../extracted_feature/mifeature.csv',index_col=0)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        lncid=torch.from_numpy(self.data[index,0]).cuda(device)
        miid=torch.from_numpy(self.data[index,1]).cuda(device)
        strlncid=torch.from_numpy(self.featurelnc.iloc[index].to_numpy()).cuda(device)
        strmiid=torch.from_numpy(self.featuremi.iloc[index].to_numpy()).cuda(device)
        label=self.data[index,4]
        return lncid,miid,strlncid.float(),strmiid.float(),label

# class Data(Dataset):
#     def __init__(self,txtpath):
#         if args.cuda_id != -1:
#             device = 'cuda:{}'.format(args.cuda_id)
#         else:
#             device = 'cpu'

#         self.data = txtpath
        
#         self.diclnc=np.load('../word2vec/lnc2vec.npy',allow_pickle=True).item()
#         self.dicmi=np.load('../word2vec/mi2vec.npy',allow_pickle=True).item()
        
#         self.featurelnc=pd.read_csv('../extracted_feature/lncfeature.csv',index_col=0)
#         self.featuremi=pd.read_csv('../extracted_feature/mifeature.csv',index_col=0)
# #         self.featurelnc=pd.read_csv('../sim_matrix/{}_matrix.csv'.format('lnccos'),index_col=0)
# #         self.featuremi=pd.read_csv('../sim_matrix/{}_matrix.csv'.format('micos'),index_col=0)
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, index):
        
#         lncseq=[self.diclnc[i] for i in self.data[index,0]]
#         miseq=[self.dicmi[i] for i in self.data[index,1]]
        
        
    
            
#         lw2v=np.array(lncseq)
#         rw2v=np.array(miseq)
        
#         lncid=torch.from_numpy(lw2v).cuda(device)
#         miid=torch.from_numpy(rw2v).cuda(device)

        
            
#         strlncid=torch.from_numpy(self.featurelnc.iloc[index].to_numpy()).cuda(device)
#         strmiid=torch.from_numpy(self.featuremi.iloc[index].to_numpy()).cuda(device)
#         label=self.data[index,4]
#         return lncid,miid,strlncid.float(),strmiid.float(),label
#         # return lncid,miid,lncid,miid,label


# %%
def train(model,trainloader,testloader,lr,ver):
        LR=lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
#         optimizer =torch.optim.SGD(model.parameters(),lr=LR, momentum=0.8)
        scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=list(range(200)),gamma = 0.98)
        with open('./logs/{}-FOLD-LOSS_{}.txt'.format(fold,ver),'w') as fp:
            
            for epoch in range(0, args.epochs):
                model.train()
                epoch_loss=0.0000
                correct, total = 0, 0
                start=time.time()
                for i, data in tqdm(enumerate(trainloader, 0)):
                    lncsams,misams,strlncsams,strmisams,labels = data
                    labels=labels.long()
                    # Zero the gradients
                    optimizer.zero_grad()
                    # Perform forward pass
                    outputs = model(misams,lncsams,strmisams,strlncsams)
                    # Compute loss
                    loss = loss_function(outputs, labels.cuda(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.cuda(device)).sum().item()
                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    optimizer.step()

                    # Print statistics
                    epoch_loss += loss.item()

                scheduler.step()
                fp.write(str(epoch_loss/(i+1)))
                fp.write('\n')
                end=time.time()
                print("time: {}".format(end-start))
                print("fold: {}  epoch:{}".format(fold+1,epoch+1))
                print("epoch loss: %.4f"%(epoch_loss/(i+1)))
                print('Accuracy for epoch %d: %.2f %%' % (epoch+1, 100.0 * correct / total))
                fp.write(str(100.0 * correct / total))
                fp.write('\n')
                correct, total = 0, 0
                model.eval()
                with torch.no_grad():
                  # Iterate over the test data and generate predictions
                    for i, data in enumerate(testloader, 0):

                        # Get inputs
                        lncsams,misams,strlncsams,strmisams,labels = data
                        labels=labels.long()
                        # Generate outputs
                        outputs = model(misams,lncsams,strmisams,strlncsams)
                        # Set total and correct
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.cuda(device)).sum().item()
                    fp.write(str(100.0 * correct / total))
                    fp.write('\n')
                    # Print accuracy
                    if epoch%3==0:
                        save_path = './Results/fold{}/model-epoch-{}_{}_{}.pth'.format(fold+1,epoch,ver,time.time())
                        torch.save(model.state_dict(), save_path)
                    print('Accuracy for test %d: %.2f %%' % (fold, 100.0 * correct / total))
    

if __name__ == '__main__':
    if args.cuda_id != -1:
        device = 'cuda:{}'.format(args.cuda_id)
    else:
        device = 'cpu'
        
    k_folds = 5
    loss_function = nn.CrossEntropyLoss()
    results = {}
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    print("Raw data has loaded...")
    raw=np.load('../Data/id_trainval_seqstr.npy',allow_pickle=True)
    
    # raw=pd.read_csv('../word2vec/mi-lncRNA_train_str_w2v.csv')
    
    dataset = Data(raw)
   
    print("Dataset has created...")
#     kfold = KFold(n_splits=k_folds, shuffle=True)

#     for fold,(train_ids, test_ids) in enumerate(kfold.split(dataset)):
#         np.save("{}_trainset.npy".format(fold),train_ids)
#         np.save("{}_testset.npy".format(fold),test_ids)

    for fold in range(5):
        train_ids=np.load("{}_trainset.npy".format(fold))
        test_ids=np.load("{}_testset.npy".format(fold))
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, sampler=test_subsampler)
        cf=config()
#         gru
        model1=AttentionMiRNATAR_v3(cf)
        model1.to(device)
        train(model1,trainloader,testloader,2e-4,'v3')
        torch.cuda.empty_cache()
        
#       cnn+attention
        model2=AttentionMiRNATAR_v2(cf)
        model2.to(device)
        train(model2,trainloader,testloader,2e-4,'v2')
        torch.cuda.empty_cache()
        
        model3=AttentionMiRNATAR_v7(cf)
        model3.to(device)
        train(model3,trainloader,testloader,2e-4,'v7')
        torch.cuda.empty_cache()


# #       sim_matrix
#         model1=AttentionMiRNATAR_v11(cf)
#         model1.to(device)
#         train(model1,trainloader,testloader,1e-4,'v11')
        
    

        

