import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class MultiScaleCNN(nn.Module):
    def __init__(self, config,flag):
        super(MultiScaleCNN, self).__init__()
        if flag==0:
            kernel_sizes=config.mi_kernel_size
            
        else:
            kernel_sizes=config.lnc_kernel_size
            
        in_channels=config.embedding_size
        out_channels=config.feature_size 
        base=config.feature_size//4
        # 定义第一个通道
        layers1=[]
        for i in range(len(kernel_sizes[0])):
            out_channels=base
            layers1.append(nn.Conv1d(in_channels , out_channels, kernel_sizes[0][i], padding=(kernel_sizes[0][i]-1) // 2))
            layers1.append(nn.ReLU(inplace=True))
            layers1.append(nn.BatchNorm1d(out_channels))
            
        self.conv1 = nn.Sequential(*layers1)
        
        in_channels=config.embedding_size
        out_channels=config.feature_size           
        
        # 定义第二个卷积层
        layers2=[]
        for i in range(len(kernel_sizes[1])):
            if i==len(kernel_sizes[1])-1:
                out_channels=base
            layers2.append(nn.Conv1d(in_channels , out_channels, kernel_sizes[1][i], padding=(kernel_sizes[1][i]-1) // 2))
            layers2.append(nn.ReLU(inplace=True))
            layers2.append(nn.BatchNorm1d(out_channels))
           
            in_channels = out_channels
            out_channels= base*(len(kernel_sizes[1])-i)
                
        self.conv2 = nn.Sequential(*layers2)
        
        in_channels=config.embedding_size
        out_channels=config.feature_size         
        # 定义第三个卷积层
        layers3=[]
        for i in range(len(kernel_sizes[2])):
            if i==len(kernel_sizes[2])-1:
                out_channels=base
            layers3.append(nn.Conv1d(in_channels , out_channels, kernel_sizes[2][i], padding=(kernel_sizes[2][i]-1) // 2))
            layers3.append(nn.ReLU(inplace=True))
            layers3.append(nn.BatchNorm1d(out_channels))
            
            in_channels = out_channels
            out_channels= base*(len(kernel_sizes[1])-i)

        self.conv3 = nn.Sequential(*layers3)
        
        in_channels=config.embedding_size
        out_channels=config.feature_size   
        
        layers4=[]
        for i in range(len(kernel_sizes[3])):
            if i==len(kernel_sizes[3])-1:
                out_channels=base
            layers4.append(nn.Conv1d(in_channels , out_channels, kernel_sizes[3][i], padding=(kernel_sizes[3][i]-1) // 2))
            layers4.append(nn.ReLU(inplace=True))
            layers4.append(nn.BatchNorm1d(out_channels))
            
            in_channels = out_channels
            out_channels= base*(len(kernel_sizes[1])-i)
        self.conv4 = nn.Sequential(*layers4)
        

        
    def forward(self, x):
        # 前向传播
        out1 = self.conv1(x)    
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        
        # 将卷积层的输出按通道方向连接起来
        out = torch.cat([out1, out2, out3,out4], dim=1)
        return out


#     MS-CNN
class AttentionMiRNATAR(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        
        self.miconvs1=nn.Conv1d(config.embedding_size , config.feature_size, 1, padding=0)
        self.miconvs2=nn.Conv1d(config.feature_size , config.embedding_size, 3, padding=1)
        self.miconvs3 =MultiScaleCNN(config,0)
       
        self.lncconvs1=nn.Conv1d(config.embedding_size , config.feature_size, 1, padding=0)
        self.lncconvs2=nn.Conv1d(config.feature_size , config.embedding_size, 3, padding=1)
        self.lncconvs3=MultiScaleCNN(config,1)
        
        self.mi_max_pool = nn.MaxPool1d(config.max_mi)
        self.lnc_max_pool = nn.MaxPool1d(config.max_lnc)
        

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)
        
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9= nn.BatchNorm1d(512)
        
   

        self.fc1 = nn.Linear(config.feature_size*2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
#         self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),out_features=config.num_class)
    def forward(self, mi,lnc,strmi,strlnc):
#         print(mi.shape)
#         print(lnc.shape)

        embed_mi = self.embedding1(mi)
        embed_mi=embed_mi.permute(0, 2, 1)
        miconv=self.miconvs3(self.miconvs2(self.miconvs1(embed_mi)))
        mi_att = self.mattention_layer(miconv.permute(0, 2, 1))
        
        embed_lnc = self.embedding2(lnc)
        embed_lnc=embed_lnc.permute(0, 2, 1)
        lncconv=self.lncconvs3(self.lncconvs2(self.lncconvs1(embed_lnc)))
        lnc_att = self.lattention_layer(lncconv.permute(0, 2, 1))
        
        mi_att_rep = torch.unsqueeze(mi_att, 2).repeat(1, 1, lncconv.shape[-1], 1)
        lnc_att_rep = torch.unsqueeze(lnc_att, 1).repeat(1, miconv.shape[-1], 1, 1) 
        att_concat = self.attention_layer(self.relu(mi_att_rep + lnc_att_rep))
        mi_attention = self.sigmoid(torch.mean(att_concat, 2).permute(0, 2, 1))
        lnc_attention = self.sigmoid(torch.mean(att_concat, 1).permute(0, 2, 1))
        alpha=0.5
        miconv = self.mi_max_pool(miconv*( mi_attention.add(0.5))).squeeze(2) 
        lncconv = self.lnc_max_pool(lncconv*( lnc_attention.add(0.5))).squeeze(2)
        

        
        feature = self.bn7(torch.cat([miconv, lncconv], dim=1))
        fully1 = self.bn8(self.leaky_relu(self.fc1(feature)))
        fully2 = self.bn9(self.leaky_relu(self.fc2(fully1)))
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

#  origin   

class AttentionMiRNATAR_v2(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v2, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.miconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.mi_kernel_size[0])
        self.miconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.mi_kernel_size[1])
        self.miconvs3 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.mi_kernel_size[2])
       
        self.lncconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.lnc_kernel_size[0])
        self.lncconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.lnc_kernel_size[1])
        self.lncconvs3 = nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.lnc_kernel_size[2])
        
        self.convsmi1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
        self.convsmi = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3,padding=1)
        self.convslnc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
        self.convslnc = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3,padding=1)
        
        
        
        self.mi_max_pool = nn.MaxPool1d(config.max_mi-sum(config.mi_kernel_size)+3)
        self.lnc_max_pool = nn.MaxPool1d(config.max_lnc-sum(config.lnc_kernel_size)+3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)

        
        self.bn1 = nn.BatchNorm1d(config.feature_size*4)
        self.bn2 = nn.BatchNorm1d(config.feature_size*2)
        self.bn3 = nn.BatchNorm1d(config.feature_size)
        self.bn4 = nn.BatchNorm1d(config.feature_size*4)
        self.bn5 = nn.BatchNorm1d(config.feature_size*2)
        self.bn6 = nn.BatchNorm1d(config.feature_size)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9= nn.BatchNorm1d(512)
        
        
        self.bn10=nn.BatchNorm1d(64)
        self.bn11=nn.BatchNorm1d(64)
        
        self.fcm=nn.Linear(226, 64)
        self.fcl=nn.Linear(226, 64)
        
        self.fc1 = nn.Linear(config.feature_size*2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
#         self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),out_features=config.num_class)
    def forward(self, mi,lnc,strmi,strlnc):
        embed_mi = self.embedding1(mi)
        embed_mi=embed_mi.permute(0, 2, 1)
#         print(embed_mi.shape)
        miconv=self.bn3(self.relu(self.miconvs3(self.bn2(self.relu(self.miconvs2(self.bn1(self.relu(self.miconvs1(embed_mi)))))))))
#         print(miconv.shape)
        mi_att = self.mattention_layer(miconv.permute(0, 2, 1))
#         print(mi_att.shape)
        embed_lnc = self.embedding2(lnc)
        embed_lnc=embed_lnc.permute(0, 2, 1)
#         print(embed_lnc.shape)
        lncconv=self.bn6(self.relu(self.lncconvs3(self.bn5(self.relu(self.lncconvs2(self.relu(self.bn4(self.lncconvs1(embed_lnc)))))))))
#         print(lncconv.shape)
        lnc_att = self.lattention_layer(lncconv.permute(0, 2, 1))
#         print(lnc_att.shape)
        mi_att_rep = torch.unsqueeze(mi_att, 2).repeat(1, 1, lncconv.shape[-1], 1)
#         print(mi_att_rep.shape)
        lnc_att_rep = torch.unsqueeze(lnc_att, 1).repeat(1, miconv.shape[-1], 1, 1) 
#         print(lnc_att_rep.shape)
        att_concat = self.attention_layer(self.relu(mi_att_rep + lnc_att_rep))
#         print(att_concat.shape)
        mi_attention = self.sigmoid(torch.mean(att_concat, 2).permute(0, 2, 1))
#         print(mi_attention.shape)
        lnc_attention = self.sigmoid(torch.mean(att_concat, 1).permute(0, 2, 1))
        alpha=0.5
        miconv = self.mi_max_pool(miconv*( mi_attention.add(0.5))).squeeze(2) 
        lncconv = self.lnc_max_pool(lncconv*( lnc_attention.add(0.5))).squeeze(2)
        
#         strmi=self.relu(self.convsmi(self.convsmi1(strmi.unsqueeze(1))))
#         strlnc=self.relu(self.convslnc(self.convslnc1(strlnc.unsqueeze(1))))
        
#         strmi=self.bn10(self.fcm(strmi.squeeze(1)))
#         strlnc=self.bn11(self.fcl(strlnc.squeeze(1)))
        
        
        feature = self.bn7(torch.cat([miconv, lncconv], dim=1))
        fully1 = self.bn8(self.leaky_relu(self.fc1(feature)))
        fully2 = self.bn9(self.leaky_relu(self.fc2(fully1)))
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict


# one gru layer    
class AttentionMiRNATAR_v3(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v3, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        
        self.migru = torch.nn.GRU(config.embedding_size, config.feature_size, 1, batch_first=True,bidirectional=True)
        self.miconvs1 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size*4, kernel_size=config.mi_kernel_size[0],padding=config.mi_kernel_size[0]//2)
        self.miconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.mi_kernel_size[1],padding=config.mi_kernel_size[1]//2)
        self.miconvs3 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.mi_kernel_size[2],padding=config.mi_kernel_size[2]//2)
       
        self.lncgru = torch.nn.GRU(config.embedding_size, config.feature_size, 1, batch_first=True,bidirectional=True)
        self.lncconvs1 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size*4, kernel_size=config.lnc_kernel_size[0],padding=config.lnc_kernel_size[0]//2)
        self.lncconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.lnc_kernel_size[1],padding=config.lnc_kernel_size[1]//2)
        self.lncconvs3 = nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.lnc_kernel_size[2],padding=config.lnc_kernel_size[2]//2)
        
        self.convsmi = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        self.convslnc = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        
        
        
        self.mi_max_pool = nn.MaxPool1d(config.max_mi-sum(config.mi_kernel_size)+3)
        self.lnc_max_pool = nn.MaxPool1d(config.max_lnc-sum(config.lnc_kernel_size)+3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)

        
        self.bn1 = nn.BatchNorm1d(config.feature_size*4)
        self.bn2 = nn.BatchNorm1d(config.feature_size*2)
        self.bn3 = nn.BatchNorm1d(config.feature_size)
        self.bn4 = nn.BatchNorm1d(config.feature_size*4)
        self.bn5 = nn.BatchNorm1d(config.feature_size*2)
        self.bn6 = nn.BatchNorm1d(config.feature_size)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9= nn.BatchNorm1d(512)
        
        
        self.bn10=nn.BatchNorm1d(32)
        self.bn11=nn.BatchNorm1d(32)
        
        self.fcm=nn.Linear(226, 32)
        self.fcl=nn.Linear(226, 32)
        
        self.fc1 = nn.Linear(config.feature_size*2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
#         self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),out_features=config.num_class)
    def forward(self, mi,lnc,strmi,strlnc):
        embed_mi = self.embedding1(mi)
        embed_mi,_=self.migru(embed_mi)
        
        embed_mi=embed_mi.permute(0, 2, 1)
#         print(embed_mi.shape)
        miconv=self.bn3(self.relu(self.miconvs3(self.bn2(self.relu(self.miconvs2(self.bn1(self.relu(self.miconvs1(embed_mi)))))))))
#         print(miconv.shape)


        embed_lnc = self.embedding2(lnc)
        embed_lnc,_=self.lncgru(embed_lnc)
        
        embed_lnc=embed_lnc.permute(0, 2, 1)
#         print(embed_lnc.shape)
        lncconv=self.bn6(self.relu(self.lncconvs3(self.bn5(self.relu(self.lncconvs2(self.relu(self.bn4(self.lncconvs1(embed_lnc)))))))))
#         print(lncconv.shape)
        miconv = self.mi_max_pool(miconv).squeeze(2) 
        lncconv = self.lnc_max_pool(lncconv).squeeze(2)
        
#         strmi=self.relu(self.convsmi(strmi.unsqueeze(1)))
#         strlnc=self.relu(self.convslnc(strlnc.unsqueeze(1)))
        
#         strmi=self.bn10(self.fcm(strmi.squeeze(1)))
#         strlnc=self.bn11(self.fcl(strlnc.squeeze(1)))
        
        
        feature = self.bn7(torch.cat([miconv, lncconv], dim=1))
        fully1 = self.bn8(self.leaky_relu(self.fc1(feature)))
        fully2 = self.bn9(self.leaky_relu(self.fc2(fully1)))
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

# one gru layer    
class AttentionMiRNATAR_v4(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v4, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        
        self.migru = torch.nn.GRU(config.embedding_size, config.feature_size, 1, batch_first=True,bidirectional=True)
        self.miconvs1 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size*4, kernel_size=config.mi_kernel_size[0],padding=config.mi_kernel_size[0]//2)
        self.miconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.mi_kernel_size[1],padding=config.mi_kernel_size[1]//2)
        self.miconvs3 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.mi_kernel_size[2],padding=config.mi_kernel_size[2]//2)
       
        self.lncgru = torch.nn.GRU(config.embedding_size, config.feature_size, 1, batch_first=True,bidirectional=True)
        self.lncconvs1 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size*4, kernel_size=config.lnc_kernel_size[0],padding=config.lnc_kernel_size[0]//2)
        self.lncconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.lnc_kernel_size[1],padding=config.lnc_kernel_size[1]//2)
        self.lncconvs3 = nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.lnc_kernel_size[2],padding=config.lnc_kernel_size[2]//2)
          
        
        self.mi_pool = nn.AdaptiveAvgPool1d(1)
        self.lnc_pool = nn.AdaptiveAvgPool1d(1)
        
        self.mi_epool = nn.AdaptiveAvgPool1d(1)
        self.lnc_epool = nn.AdaptiveAvgPool1d(1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)

        
        self.bn1 = nn.BatchNorm1d(config.feature_size*4)
        self.bn2 = nn.BatchNorm1d(config.feature_size*2)
        self.bn3 = nn.BatchNorm1d(config.feature_size)
        self.bn4 = nn.BatchNorm1d(config.feature_size*4)
        self.bn5 = nn.BatchNorm1d(config.feature_size*2)
        self.bn6 = nn.BatchNorm1d(config.feature_size)
        
        self.out = nn.Linear(in_features=config.feature_size*6,out_features=config.num_class)
        
    def forward(self, mi,lnc,strmi,strlnc):
        embed_mi = self.embedding1(mi)
        embed_mi,_=self.migru(embed_mi)
        
        embed_mpool=self.mi_epool(embed_mi)
        
        embed_mi=embed_mi.permute(0, 2, 1)
        miconv=self.bn3(self.relu(self.miconvs3(self.bn2(self.relu(self.miconvs2(self.bn1(self.relu(self.miconvs1(embed_mi)))))))))
        mi_att = self.mattention_layer(miconv.permute(0, 2, 1))
        embed_lnc = self.embedding2(lnc)
        embed_lnc,_=self.lncgru(embed_lnc)
        
        embed_lpool=self.mi_epool(embed_mi)
        
        embed_lnc=embed_lnc.permute(0, 2, 1)
        lncconv=self.bn6(self.relu(self.lncconvs3(self.bn5(self.relu(self.lncconvs2(self.relu(self.bn4(self.lncconvs1(embed_lnc)))))))))
        lnc_att = self.lattention_layer(lncconv.permute(0, 2, 1))
        mi_att_rep = torch.unsqueeze(mi_att, 2).repeat(1, 1, lncconv.shape[-1], 1)
        lnc_att_rep = torch.unsqueeze(lnc_att, 1).repeat(1, miconv.shape[-1], 1, 1) 
        att_concat = self.attention_layer(self.relu(mi_att_rep + lnc_att_rep))
        mi_attention = self.sigmoid(torch.mean(att_concat, 2).permute(0, 2, 1))
        lnc_attention = self.sigmoid(torch.mean(att_concat, 1).permute(0, 2, 1))
        alpha=0.5
        miconv = self.mi_pool(miconv*( mi_attention.add(0.5))).squeeze(2) 
        lncconv = self.lnc_pool(lncconv*( lnc_attention.add(0.5))).squeeze(2)
        
        
        feature = torch.cat([miconv, lncconv,embed_mpool,embed_lpool], dim=1)
        predict = self.out(feature)
        return predict

    

#     two liner layer
class AttentionMiRNATAR_v5(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v5, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.miconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.mi_kernel_size[0])
        self.miconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.mi_kernel_size[1])
        self.miconvs3 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.mi_kernel_size[2])
       
        self.lncconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.lnc_kernel_size[0])
        self.lncconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.lnc_kernel_size[1])
        self.lncconvs3 = nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.lnc_kernel_size[2])
        
        self.convsmi1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
        self.convsmi = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3,padding=1)
        self.convslnc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
        self.convslnc = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3,padding=1)
        self.fcml=nn.Linear(226*2, 128)
        
        
        self.mi_max_pool = nn.MaxPool1d(config.max_mi-sum(config.mi_kernel_size)+3)
        self.lnc_max_pool = nn.MaxPool1d(config.max_lnc-sum(config.lnc_kernel_size)+3)
        
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)

        
        self.bn1 = nn.BatchNorm1d(config.feature_size*4)
        self.bn2 = nn.BatchNorm1d(config.feature_size*2)
        self.bn3 = nn.BatchNorm1d(config.feature_size)
        self.bn4 = nn.BatchNorm1d(config.feature_size*4)
        self.bn5 = nn.BatchNorm1d(config.feature_size*2)
        self.bn6 = nn.BatchNorm1d(config.feature_size)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9= nn.BatchNorm1d(512)
        
        
        self.bn10=nn.BatchNorm1d(128)
   
        self.fc1 = nn.Linear(config.feature_size*4, 512)
#         self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
    def forward(self, mi,lnc,strmi,strlnc):
        embed_mi = self.embedding1(mi)
        embed_mi=embed_mi.permute(0, 2, 1)
#         print(embed_mi.shape)
        miconv=self.bn3(self.relu(self.miconvs3(self.bn2(self.relu(self.miconvs2(self.bn1(self.relu(self.miconvs1(embed_mi)))))))))
#         print(miconv.shape)
        mi_att = self.mattention_layer(miconv.permute(0, 2, 1))
#         print(mi_att.shape)
        embed_lnc = self.embedding2(lnc)
        embed_lnc=embed_lnc.permute(0, 2, 1)
#         print(embed_lnc.shape)
        lncconv=self.bn6(self.relu(self.lncconvs3(self.bn5(self.relu(self.lncconvs2(self.relu(self.bn4(self.lncconvs1(embed_lnc)))))))))
#         print(lncconv.shape)
        lnc_att = self.lattention_layer(lncconv.permute(0, 2, 1))
#         print(lnc_att.shape)
        mi_att_rep = torch.unsqueeze(mi_att, 2).repeat(1, 1, lncconv.shape[-1], 1)
#         print(mi_att_rep.shape)
        lnc_att_rep = torch.unsqueeze(lnc_att, 1).repeat(1, miconv.shape[-1], 1, 1) 
#         print(lnc_att_rep.shape)
        att_concat = self.attention_layer(self.relu(mi_att_rep + lnc_att_rep))
#         print(att_concat.shape)
        mi_attention = self.sigmoid(torch.mean(att_concat, 2).permute(0, 2, 1))
#         print(mi_attention.shape)
        lnc_attention = self.sigmoid(torch.mean(att_concat, 1).permute(0, 2, 1))
        alpha=0.5
        miconv = self.mi_max_pool(miconv*( mi_attention.add(0.5))).squeeze(2) 
        lncconv = self.lnc_max_pool(lncconv*( lnc_attention.add(0.5))).squeeze(2)
        
        strmi=self.relu(self.convsmi(self.convsmi1(strmi.unsqueeze(1))))
        strlnc=self.relu(self.convslnc(self.convslnc1(strlnc.unsqueeze(1))))
        
        exfeature=self.bn10(self.leaky_relu(self.fcml(torch.cat([strmi.squeeze(1),strlnc.squeeze(1)],dim=1))))
        
        
        
        
        feature = self.bn7(torch.cat([miconv, lncconv], dim=1))
                            
        fully1 = self.bn8(self.leaky_relu(self.fc1(torch.cat([feature, exfeature], dim=1))))
#         fully2 = self.bn9(self.leaky_relu(self.fc2(fully1)))
        fully3 = self.leaky_relu(self.fc3(fully1))
        predict = self.out(fully3)
        return predict

#     no attention
class AttentionMiRNATAR_v6(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v6, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.miconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.mi_kernel_size[0])
        self.miconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.mi_kernel_size[1])
        self.miconvs3 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.mi_kernel_size[2])
       
        self.lncconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.lnc_kernel_size[0])
        self.lncconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.lnc_kernel_size[1])
        self.lncconvs3 = nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.lnc_kernel_size[2])
        
        self.convsmi1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
        self.convsmi = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3,padding=1)
        self.convslnc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
        self.convslnc = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3,padding=1)
        self.fcml=nn.Linear(226*2, 128)
        
        
        self.mi_max_pool = nn.MaxPool1d(config.max_mi-sum(config.mi_kernel_size)+3)
        self.lnc_max_pool = nn.MaxPool1d(config.max_lnc-sum(config.lnc_kernel_size)+3)
        
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)

        
        self.bn1 = nn.BatchNorm1d(config.feature_size*4)
        self.bn2 = nn.BatchNorm1d(config.feature_size*2)
        self.bn3 = nn.BatchNorm1d(config.feature_size)
        self.bn4 = nn.BatchNorm1d(config.feature_size*4)
        self.bn5 = nn.BatchNorm1d(config.feature_size*2)
        self.bn6 = nn.BatchNorm1d(config.feature_size)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9= nn.BatchNorm1d(512)
        
        
        self.bn10=nn.BatchNorm1d(128)
   
        
        
        self.fc1 = nn.Linear(config.feature_size*4, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
    def forward(self, mi,lnc,strmi,strlnc):
        embed_mi = self.embedding1(mi)
        embed_mi=embed_mi.permute(0, 2, 1)
        miconv=self.bn3(self.relu(self.miconvs3(self.bn2(self.relu(self.miconvs2(self.bn1(self.relu(self.miconvs1(embed_mi)))))))))

        embed_lnc = self.embedding2(lnc)
        embed_lnc=embed_lnc.permute(0, 2, 1)
        lncconv=self.bn6(self.relu(self.lncconvs3(self.bn5(self.relu(self.lncconvs2(self.relu(self.bn4(self.lncconvs1(embed_lnc)))))))))

        miconv = self.mi_max_pool(miconv).squeeze(2) 
        lncconv = self.lnc_max_pool(lncconv).squeeze(2)
        
        strmi=self.relu(self.convsmi(self.convsmi1(strmi.unsqueeze(1))))
        strlnc=self.relu(self.convslnc(self.convslnc1(strlnc.unsqueeze(1))))
        
        exfeature=self.bn10(self.leaky_relu(self.fcml(torch.cat([strmi.squeeze(1),strlnc.squeeze(1)],dim=1))))
        
        feature = self.bn7(torch.cat([miconv, lncconv], dim=1))
                            
        fully1 = self.bn8(self.leaky_relu(self.fc1(torch.cat([feature, exfeature], dim=1))))
#         fully2 = self.bn9(self.leaky_relu(self.fc2(fully1)))
        fully3 = self.leaky_relu(self.fc3(fully1))
        predict = self.out(fully3)
        return predict



class AttentionMiRNATAR_v7(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v7, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        
        
        self.miconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.mi_kernel_size[0],padding=config.mi_kernel_size[0]//2)
        self.miconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*3, kernel_size=config.mi_kernel_size[1],padding=config.mi_kernel_size[1]//2)
        self.miconvs3 =nn.Conv1d(in_channels=config.feature_size*3, out_channels=config.feature_size*2, kernel_size=config.mi_kernel_size[2],padding=config.mi_kernel_size[2]//2)
        
        self.migru = torch.nn.GRU(config.feature_size*2, config.feature_size, 1, batch_first=True,bidirectional=True)
       
        
        self.lncconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.lnc_kernel_size[0],padding=config.lnc_kernel_size[0]//2)
        self.lncconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*3, kernel_size=config.lnc_kernel_size[1],padding=config.lnc_kernel_size[1]//2)
        self.lncconvs3 = nn.Conv1d(in_channels=config.feature_size*3, out_channels=config.feature_size*2, kernel_size=config.lnc_kernel_size[2],padding=config.lnc_kernel_size[2]//2)
        
        self.lncgru = torch.nn.GRU(config.feature_size*2, config.feature_size, 1, batch_first=True,bidirectional=True)
        
        self.convsmi = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        self.convslnc = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        
        
        
        self.mi_max_pool = nn.MaxPool1d(config.max_mi-sum(config.mi_kernel_size)+3)
        self.lnc_max_pool = nn.MaxPool1d(config.max_lnc-sum(config.lnc_kernel_size)+3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)

        
        self.bn1 = nn.BatchNorm1d(config.feature_size*4)
        self.bn2 = nn.BatchNorm1d(config.feature_size*3)
        self.bn3 = nn.BatchNorm1d(config.feature_size*2)
        self.bn4 = nn.BatchNorm1d(config.feature_size*4)
        self.bn5 = nn.BatchNorm1d(config.feature_size*3)
        self.bn6 = nn.BatchNorm1d(config.feature_size*2)
        self.bn7 = nn.BatchNorm1d(config.feature_size*4)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9= nn.BatchNorm1d(512)
        
        
        self.bn10=nn.BatchNorm1d(32)
        self.bn11=nn.BatchNorm1d(32)
        
        self.fcm=nn.Linear(226, 32)
        self.fcl=nn.Linear(226, 32)
        
        self.fc1 = nn.Linear(config.feature_size*4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
#         self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),out_features=config.num_class)
    def forward(self, mi,lnc,strmi,strlnc):
        embed_mi = self.embedding1(mi)
       
        
        embed_mi=embed_mi.permute(0, 2, 1)
#         print(embed_mi.shape)
        miconv=self.bn3(self.relu(self.miconvs3(self.bn2(self.relu(self.miconvs2(self.bn1(self.relu(self.miconvs1(embed_mi)))))))))
#         print(miconv.shape)

        miout,mihn=self.migru(miconv.permute(0, 2, 1))
    
        x = []
        for i in range(mihn.size(0)):
              x.append(mihn[i, :, :])
        mihn = torch.cat(x, dim=-1)

        embed_lnc = self.embedding2(lnc)
        
        
        embed_lnc=embed_lnc.permute(0, 2, 1)
#         print(embed_lnc.shape)
        lncconv=self.bn6(self.relu(self.lncconvs3(self.bn5(self.relu(self.lncconvs2(self.relu(self.bn4(self.lncconvs1(embed_lnc)))))))))
       
        
        lncout,lnchn=self.lncgru(lncconv.permute(0, 2, 1))
#         print(lnchn.shape)
        x = []
        for i in range(lnchn.size(0)):
              x.append(lnchn[i, :, :])
        lnchn = torch.cat(x, dim=-1)

        
#         strmi=self.relu(self.convsmi(strmi.unsqueeze(1)))
#         strlnc=self.relu(self.convslnc(strlnc.unsqueeze(1)))
        
#         strmi=self.bn10(self.fcm(strmi.squeeze(1)))
#         strlnc=self.bn11(self.fcl(strlnc.squeeze(1)))
        
        
        feature = self.bn7(torch.cat([mihn, lnchn], dim=1))
        fully1 = self.bn8(self.leaky_relu(self.fc1(feature)))
        fully2 = self.bn9(self.leaky_relu(self.fc2(fully1)))
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

class AttentionMiRNATAR_v8(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v8, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        
        self.migru = torch.nn.GRU(config.embedding_size, config.feature_size, 1, batch_first=True,bidirectional=True)
        self.miconvs1 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size*4, kernel_size=config.mi_kernel_size[0],padding=config.mi_kernel_size[0]//2)
        self.miconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.mi_kernel_size[1],padding=config.mi_kernel_size[1]//2)
        self.miconvs3 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.mi_kernel_size[2],padding=config.mi_kernel_size[2]//2)
       
        self.lncgru = torch.nn.GRU(config.embedding_size, config.feature_size, 1, batch_first=True,bidirectional=True)
        self.lncconvs1 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size*4, kernel_size=config.lnc_kernel_size[0],padding=config.lnc_kernel_size[0]//2)
        self.lncconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.lnc_kernel_size[1],padding=config.lnc_kernel_size[1]//2)
        self.lncconvs3 = nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.lnc_kernel_size[2],padding=config.lnc_kernel_size[2]//2)
        
        self.convsmi = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        self.convslnc = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        
        
        
        self.mi_max_pool = nn.MaxPool1d(config.max_mi-sum(config.mi_kernel_size)+3)
        self.lnc_max_pool = nn.MaxPool1d(config.max_lnc-sum(config.lnc_kernel_size)+3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)

        
        self.bn1 = nn.BatchNorm1d(config.feature_size*4)
        self.bn2 = nn.BatchNorm1d(config.feature_size*2)
        self.bn3 = nn.BatchNorm1d(config.feature_size)
        self.bn4 = nn.BatchNorm1d(config.feature_size*4)
        self.bn5 = nn.BatchNorm1d(config.feature_size*2)
        self.bn6 = nn.BatchNorm1d(config.feature_size)
        self.bn7 = nn.BatchNorm1d(448)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9= nn.BatchNorm1d(512)
        
        
        self.bn10=nn.BatchNorm1d(32)
        self.bn11=nn.BatchNorm1d(32)
        
        self.fcm=nn.Linear(226, 32)
        self.fcl=nn.Linear(226, 32)
        
        self.fc1 = nn.Linear(config.feature_size*7, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
#         self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),out_features=config.num_class)
    def forward(self, mi,lnc,strmi,strlnc):
        embed_mi = self.embedding1(mi)
        embed_mi,_=self.migru(embed_mi)
        
        embed_mi=embed_mi.permute(0, 2, 1)
#         print(embed_mi.shape)
        miconv=self.bn3(self.relu(self.miconvs3(self.bn2(self.relu(self.miconvs2(self.bn1(self.relu(self.miconvs1(embed_mi)))))))))
#         print(miconv.shape)
        
        miconv=torch.cat([miconv, embed_mi], dim=1)


        embed_lnc = self.embedding2(lnc)
        embed_lnc,_=self.lncgru(embed_lnc)
        
        embed_lnc=embed_lnc.permute(0, 2, 1)
#         print(embed_lnc.shape)
        lncconv=self.bn6(self.relu(self.lncconvs3(self.bn5(self.relu(self.lncconvs2(self.relu(self.bn4(self.lncconvs1(embed_lnc)))))))))
    
        lncconv=torch.cat([lncconv, embed_lnc], dim=1)
#         print(lncconv.shape)
        miconv = self.mi_max_pool(miconv).squeeze(2) 
        lncconv = self.lnc_max_pool(lncconv).squeeze(2)
        
        strmi=self.relu(self.convsmi(strmi.unsqueeze(1)))
        strlnc=self.relu(self.convslnc(strlnc.unsqueeze(1)))
        
        strmi=self.bn10(self.fcm(strmi.squeeze(1)))
        strlnc=self.bn11(self.fcl(strlnc.squeeze(1)))
        
        
        feature = self.bn7(torch.cat([miconv, lncconv,strmi,strlnc], dim=1))
        fully1 = self.bn8(self.leaky_relu(self.fc1(feature)))
        fully2 = self.bn9(self.leaky_relu(self.fc2(fully1)))
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

    

class RSBU_CW(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)
 
    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)  
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding=torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)
 
        result = x + input
        return result


class AttentionMiRNATAR_v10(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v9, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        
        self.gru1 = torch.nn.GRU(config.embedding_size, 64, 1, batch_first=True,bidirectional=True)
        self.gru2 = torch.nn.GRU(config.embedding_size, 64, 1, batch_first=True,bidirectional=True)

        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear6_8 = nn.Linear(in_features=256, out_features=128)
        self.linear8_4 =nn.Linear(in_features=128, out_features=64)
        self.linear4_2 = nn.Linear(in_features=64, out_features=32)
        self.output_center_pos = nn.Linear(in_features=32, out_features=1)
        self.output_width = nn.Linear(in_features=32, out_features=1)
 
        self.linear1 = nn.Linear(in_features=1024, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=128)
        self.output_class = nn.Linear(in_features=128, out_features=2)
 
    def forward(self, input1,input2,input3,input4):  # 1*256
        x1 = self.embedding1(input1)
        x1,_=self.gru1(x1)
        x1=x1.permute(0, 2, 1)
        x1 = RSBU_CW(in_channels=128, out_channels=128, kernel_size=3, down_sample=True).cuda()(x1)  # 4*64
        x1 = RSBU_CW(in_channels=128, out_channels=128, kernel_size=3, down_sample=False).cuda()(x1)  # 4*64
        x1 = RSBU_CW(in_channels=128, out_channels=256, kernel_size=3, down_sample=True).cuda()(x1)  # 8*32
        x1 = RSBU_CW(in_channels=256, out_channels=256, kernel_size=3, down_sample=False).cuda()(x1)  # 8*32
        x1 = RSBU_CW(in_channels=256, out_channels=512, kernel_size=3, down_sample=True).cuda()(x1)  # 16*16
        x1 = RSBU_CW(in_channels=512, out_channels=512, kernel_size=3, down_sample=False).cuda()(x1)  # 16*16
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        gap1 = self.global_average_pool(x1)  # 16*1
        gap1 = self.flatten(gap1)  # 1*16

        x2 = self.embedding2(input2)
        x2,_=self.gru2(x2)
        x2=x2.permute(0, 2, 1)
        x2= RSBU_CW(in_channels=128, out_channels=128, kernel_size=3, down_sample=True).cuda()(x2)  # 4*64
        x2= RSBU_CW(in_channels=128, out_channels=128, kernel_size=3, down_sample=False).cuda()(x2)  # 4*64
        x2= RSBU_CW(in_channels=128, out_channels=256, kernel_size=3, down_sample=True).cuda()(x2)  # 8*32
        x2= RSBU_CW(in_channels=256, out_channels=256, kernel_size=3, down_sample=False).cuda()(x2)  # 8*32
        x2= RSBU_CW(in_channels=256, out_channels=512, kernel_size=3, down_sample=True).cuda()(x2)  # 16*16
        x2= RSBU_CW(in_channels=512, out_channels=512, kernel_size=3, down_sample=False).cuda()(x2)  # 16*16
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        gap2 = self.global_average_pool(x2)  # 16*1
        gap2 = self.flatten(gap2)  # 1*16
        
        gap=torch.cat([gap1,gap2], dim=1)
        
        
        linear1 =self.bn5(self.linear3(self.bn4(self.linear2(self.bn3(self.linear1(gap))))))  # 1*8
        output_class = self.output_class(linear1)  # 1*2
 
        return output_class

class AttentionMiRNATAR_v9(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v9, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        self.embedding2 = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,padding_idx=0)
        
        self.gru1 = torch.nn.GRU(config.embedding_size, 64, 1, batch_first=True,bidirectional=True)
        self.gru2 = torch.nn.GRU(config.embedding_size, 64, 1, batch_first=True,bidirectional=True)

        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear6_8 = nn.Linear(in_features=256, out_features=128)
        self.linear8_4 =nn.Linear(in_features=128, out_features=64)
        self.linear4_2 = nn.Linear(in_features=64, out_features=32)
        self.output_center_pos = nn.Linear(in_features=32, out_features=1)
        self.output_width = nn.Linear(in_features=32, out_features=1)
 
        self.linear1 = nn.Linear(in_features=1024, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=128)
        self.output_class = nn.Linear(in_features=128, out_features=2)
 
    def forward(self, input1,input2,input3,input4):  # 1*256
        x1 = self.embedding1(input1)
        x1,_=self.gru1(x1)
        x1=x1.permute(0, 2, 1)
        x1 = RSBU_CW(in_channels=128, out_channels=128, kernel_size=3, down_sample=True).cuda()(x1)  # 4*64
        x1 = RSBU_CW(in_channels=128, out_channels=128, kernel_size=3, down_sample=False).cuda()(x1)  # 4*64
        x1 = RSBU_CW(in_channels=128, out_channels=256, kernel_size=3, down_sample=True).cuda()(x1)  # 8*32
        x1 = RSBU_CW(in_channels=256, out_channels=256, kernel_size=3, down_sample=False).cuda()(x1)  # 8*32
        x1 = RSBU_CW(in_channels=256, out_channels=512, kernel_size=3, down_sample=True).cuda()(x1)  # 16*16
        x1 = RSBU_CW(in_channels=512, out_channels=512, kernel_size=3, down_sample=False).cuda()(x1)  # 16*16
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        gap1 = self.global_average_pool(x1)  # 16*1
        gap1 = self.flatten(gap1)  # 1*16

        x2 = self.embedding2(input2)
        x2,_=self.gru2(x2)
        x2=x2.permute(0, 2, 1)
        x2= RSBU_CW(in_channels=128, out_channels=128, kernel_size=3, down_sample=True).cuda()(x2)  # 4*64
        x2= RSBU_CW(in_channels=128, out_channels=128, kernel_size=3, down_sample=False).cuda()(x2)  # 4*64
        x2= RSBU_CW(in_channels=128, out_channels=256, kernel_size=3, down_sample=True).cuda()(x2)  # 8*32
        x2= RSBU_CW(in_channels=256, out_channels=256, kernel_size=3, down_sample=False).cuda()(x2)  # 8*32
        x2= RSBU_CW(in_channels=256, out_channels=512, kernel_size=3, down_sample=True).cuda()(x2)  # 16*16
        x2= RSBU_CW(in_channels=512, out_channels=512, kernel_size=3, down_sample=False).cuda()(x2)  # 16*16
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        gap2 = self.global_average_pool(x2)  # 16*1
        gap2 = self.flatten(gap2)  # 1*16
        
        gap=torch.cat([gap1,gap2], dim=1)
        
        
        linear1 =self.bn5(self.linear3(self.bn4(self.linear2(self.bn3(self.linear1(gap))))))  # 1*8
        output_class = self.output_class(linear1)  # 1*2
 
        return output_class


class AttentionMiRNATAR_v10(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v10, self).__init__()
        self.miconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.mi_kernel_size[0])
        self.miconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.mi_kernel_size[1])
        self.miconvs3 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.mi_kernel_size[2])
       
        self.lncconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.lnc_kernel_size[0])
        self.lncconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.lnc_kernel_size[1])
        self.lncconvs3 = nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.lnc_kernel_size[2])
        
        self.convsmi1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
        self.convsmi = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3,padding=1)
        self.convslnc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
        self.convslnc = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3,padding=1)
        
        
        
        self.mi_max_pool = nn.MaxPool1d(config.max_mi-sum(config.mi_kernel_size)+3)
        self.lnc_max_pool = nn.MaxPool1d(config.max_lnc-sum(config.lnc_kernel_size)+3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)

        
        self.bn1 = nn.BatchNorm1d(config.feature_size*4)
        self.bn2 = nn.BatchNorm1d(config.feature_size*2)
        self.bn3 = nn.BatchNorm1d(config.feature_size)
        self.bn4 = nn.BatchNorm1d(config.feature_size*4)
        self.bn5 = nn.BatchNorm1d(config.feature_size*2)
        self.bn6 = nn.BatchNorm1d(config.feature_size)
        self.bn7 = nn.BatchNorm1d(128+128)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9= nn.BatchNorm1d(512)
        
        
        self.bn10=nn.BatchNorm1d(64)
        self.bn11=nn.BatchNorm1d(64)
        
        self.fcm=nn.Linear(226, 64)
        self.fcl=nn.Linear(226, 64)
        
        self.fc1 = nn.Linear(config.feature_size*4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
#         self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),out_features=config.num_class)
    def forward(self, mi,lnc,strmi,strlnc):
        embed_mi = mi.float()
        embed_mi=embed_mi.permute(0, 2, 1)
#         print(embed_mi.shape)
        miconv=self.bn3(self.relu(self.miconvs3(self.bn2(self.relu(self.miconvs2(self.bn1(self.relu(self.miconvs1(embed_mi)))))))))
#         print(miconv.shape)
        mi_att = self.mattention_layer(miconv.permute(0, 2, 1))
#         print(mi_att.shape)


        embed_lnc = lnc.float()
        embed_lnc=embed_lnc.permute(0, 2, 1)
#         print(embed_lnc.shape)
        lncconv=self.bn6(self.relu(self.lncconvs3(self.bn5(self.relu(self.lncconvs2(self.relu(self.bn4(self.lncconvs1(embed_lnc)))))))))
#         print(lncconv.shape)
        lnc_att = self.lattention_layer(lncconv.permute(0, 2, 1))
#         print(lnc_att.shape)
        mi_att_rep = torch.unsqueeze(mi_att, 2).repeat(1, 1, lncconv.shape[-1], 1)
#         print(mi_att_rep.shape)
        lnc_att_rep = torch.unsqueeze(lnc_att, 1).repeat(1, miconv.shape[-1], 1, 1) 
#         print(lnc_att_rep.shape)
        att_concat = self.attention_layer(self.relu(mi_att_rep + lnc_att_rep))
#         print(att_concat.shape)
        mi_attention = self.sigmoid(torch.mean(att_concat, 2).permute(0, 2, 1))
#         print(mi_attention.shape)
        lnc_attention = self.sigmoid(torch.mean(att_concat, 1).permute(0, 2, 1))
        alpha=0.5
        
        miconv = self.mi_max_pool(miconv*( mi_attention.add(0.5))).squeeze(2) 
        lncconv = self.lnc_max_pool(lncconv*( lnc_attention.add(0.5))).squeeze(2)
        
        strmi=self.relu(self.convsmi(self.convsmi1(strmi.unsqueeze(1))))
        strlnc=self.relu(self.convslnc(self.convslnc1(strlnc.unsqueeze(1))))
        
        strmi=self.bn10(self.fcm(strmi.squeeze(1)))
        strlnc=self.bn11(self.fcl(strlnc.squeeze(1)))
        
        
        feature = self.bn7(torch.cat([miconv, lncconv,strmi,strlnc], dim=1))
        fully1 = self.bn8(self.leaky_relu(self.fc1(feature)))
        fully2 = self.bn9(self.leaky_relu(self.fc2(fully1)))
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

    


class AttentionMiRNATAR_v11(nn.Module):
    def __init__(self, config):
        super(AttentionMiRNATAR_v11, self).__init__()
        
        self.migru = torch.nn.GRU(config.embedding_size, config.feature_size, 1, batch_first=True,bidirectional=True)
        self.miconvs1 =nn.Conv1d(in_channels=config.embedding_size, out_channels=config.feature_size*4, kernel_size=config.mi_kernel_size[0],padding=config.mi_kernel_size[0]//2)
        self.miconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.mi_kernel_size[1],padding=config.mi_kernel_size[1]//2)
        self.miconvs3 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.mi_kernel_size[2],padding=config.mi_kernel_size[2]//2)
       
        self.lncgru = torch.nn.GRU(config.embedding_size, config.feature_size, 1, batch_first=True,bidirectional=True)
        self.lncconvs1 =nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size*4, kernel_size=config.lnc_kernel_size[0],padding=config.lnc_kernel_size[0]//2)
        self.lncconvs2 =nn.Conv1d(in_channels=config.feature_size*4, out_channels=config.feature_size*2, kernel_size=config.lnc_kernel_size[1],padding=config.lnc_kernel_size[1]//2)
        self.lncconvs3 = nn.Conv1d(in_channels=config.feature_size*2, out_channels=config.feature_size, kernel_size=config.lnc_kernel_size[2],padding=config.lnc_kernel_size[2]//2)
        
        self.convsmi = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        self.convslnc = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        
        
        
        self.mi_max_pool = nn.MaxPool1d(config.max_mi-sum(config.mi_kernel_size)+3)
        self.lnc_max_pool = nn.MaxPool1d(config.max_lnc-sum(config.lnc_kernel_size)+3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.lattention_layer = nn.Linear(config.feature_size,config.feature_size)
        self.mattention_layer = nn.Linear(config.feature_size,config.feature_size)

        
        self.bn1 = nn.BatchNorm1d(config.feature_size*4)
        self.bn2 = nn.BatchNorm1d(config.feature_size*2)
        self.bn3 = nn.BatchNorm1d(config.feature_size)
        self.bn4 = nn.BatchNorm1d(config.feature_size*4)
        self.bn5 = nn.BatchNorm1d(config.feature_size*2)
        self.bn6 = nn.BatchNorm1d(config.feature_size)
        self.bn7 = nn.BatchNorm1d(128+64)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9= nn.BatchNorm1d(512)
        
        
        self.bn10=nn.BatchNorm1d(32)
        self.bn11=nn.BatchNorm1d(32)
        
        self.fcm=nn.Linear(226, 32)
        self.fcl=nn.Linear(226, 32)
        
        self.fc1 = nn.Linear(config.feature_size*3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2)
#         self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),out_features=config.num_class)
    def forward(self, mi,lnc,strmi,strlnc):
        embed_mi = mi.float()
        
        embed_mi=embed_mi.permute(0, 2, 1)
#         print(embed_mi.shape)
        miconv=self.bn3(self.relu(self.miconvs3(self.bn2(self.relu(self.miconvs2(self.bn1(self.relu(self.miconvs1(embed_mi)))))))))
#         print(miconv.shape)


        embed_lnc = lnc.float()
        embed_lnc,_=self.lncgru(embed_lnc)
        
        embed_lnc=embed_lnc.permute(0, 2, 1)
#         print(embed_lnc.shape)
        lncconv=self.bn6(self.relu(self.lncconvs3(self.bn5(self.relu(self.lncconvs2(self.relu(self.bn4(self.lncconvs1(embed_lnc)))))))))
#         print(lncconv.shape)
        miconv = self.mi_max_pool(miconv).squeeze(2) 
        lncconv = self.lnc_max_pool(lncconv).squeeze(2)
        
        strmi=self.relu(self.convsmi(strmi.unsqueeze(1)))
        strlnc=self.relu(self.convslnc(strlnc.unsqueeze(1)))
        
        strmi=self.bn10(self.fcm(strmi.squeeze(1)))
        strlnc=self.bn11(self.fcl(strlnc.squeeze(1)))
        
        
        feature = self.bn7(torch.cat([miconv, lncconv,strmi,strlnc], dim=1))
        fully1 = self.bn8(self.leaky_relu(self.fc1(feature)))
        fully2 = self.bn9(self.leaky_relu(self.fc2(fully1)))
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict
