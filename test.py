from sklearn.metrics import accuracy_score
import numpy as np
import catboost
from catboost import CatBoostClassifier

from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,average_precision_score,roc_curve

from bayes_opt import BayesianOptimization

from matplotlib import pyplot as plt
import seaborn as sns
import random
import matplotlib.patches as mpatches

def evalue(s,l):
   
    p_new=[]

    l=np.array(l).astype('float')


    FPR, recall, thresholds1 = roc_curve(l,s, pos_label=1)
    gmean=np.sqrt(recall*(1-FPR))
    index=np.argmax(gmean)
    th1=thresholds1[index]
    print(th1)


    for i in s:
        if i<th1:
            p_new.append(0)
        else:
            p_new.append(1)
    # s=s3

    p=p_new

    tn,fp,fn,tp = confusion_matrix(l, p).ravel() 
    print(tn,fp,fn,tp)
    pp = tp+fn
    pn = tn+fp
    sensitivity = tp / pp
    specificity = tn / pn
    precision = tp / (tp + fp)
    acc = (tp+tn) / (pp+pn)
    mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    print("F1-Score:{:.4f}".format(f1_score(l,p)))
    print("AUC:{:.4f}".format(roc_auc_score(l,s)))
    print("ACC:{:.4f}".format(acc))
    print("Sn:{:.4f}".format(sensitivity))
    print("Sp:{:.4f}".format(specificity))
    print("Pre:{:.4f}".format(precision))
    print("Mcc:{:.4f}".format(mcc))
    print("PRC:{:.4f}".format(average_precision_score(l,s)))    
    
    return f1_score(l,p),roc_auc_score(l,s),acc,sensitivity,specificity,precision,mcc,average_precision_score(l,s)


params={
    'loss_function': 'CrossEntropy', # 损失函数，取值RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Multiclass, MultiClassOneVsAll, MAPE, Poisson。默认Logloss。
    'custom_loss': 'AUC', # 训练过程中计算显示的损失函数，取值Logloss、CrossEntropy、Precision、Recall、F、F1、BalancedAccuracy、AUC等等
    'eval_metric': 'AUC', # 用于过度拟合检测和最佳模型选择的指标，取值范围同custom_loss
    'iterations': 1000, # 最大迭代次数，默认500. 别名：num_boost_round, n_estimators, num_trees
    'learning_rate': 0.005, # 学习速率,默认0.03 别名：eta
    'random_seed': 42, # 训练的随机种子，别名：random_state，42
    'l2_leaf_reg': 0, # l2正则项，别名：reg_lambda
    'bootstrap_type': 'MVS', # 确定抽样时的样本权重，取值Bayesian、Bernoulli(伯努利实验)、MVS(仅支持cpu)、Poisson(仅支持gpu)、No（取值为No时，每棵树为简单随机抽样）;默认值GPU下为Bayesian、CPU下为MVS
#     'bagging_temperature': 0,  # bootstrap_type=Bayesian时使用,取值为1时采样权重服从指数分布；取值为0时所有采样权重均等于1。取值范围[0，inf)，值越大、bagging就越激进
    'subsample': 0.4, # 样本采样比率（行采样）
    'sampling_frequency': 'PerTree', # 采样频率，取值PerTree（在构建每棵新树之前采样）、PerTreeLevel（默认值，在子树的每次分裂之前采样）；仅支持CPU
    'use_best_model': True, # 让模型使用效果最优的子树棵树/迭代次数，使用验证集的最优效果对应的迭代次数（eval_metric：评估指标，eval_set：验证集数据），布尔类型可取值0，1（取1时要求设置验证集数据）
    'best_model_min_trees': 50, # 最少子树棵树,和use_best_model一起使用
    'depth':4, # 树深，默认值6
    'grow_policy': 'SymmetricTree', # 子树生长策略，取值SymmetricTree（默认值，对称树）、Depthwise（整层生长，同xgb）、Lossguide（叶子结点生长，同lgb）
    'min_data_in_leaf': 500, # 叶子结点最小样本量
#     'max_leaves': 12, # 最大叶子结点数量
    'one_hot_max_size': 6, # 对唯一值数量<one_hot_max_size的类别型特征使用one-hot编码
    'rsm': 0.2, # 列采样比率，别名colsample_bylevel 取值（0，1],默认值1
    'nan_mode': 'Max', # 缺失值处理方法，取值Forbidden（不支持缺失值，输入包含缺失时会报错）、Min（处理为该列的最小值，比原最小值更小）、Max（同理）
    'input_borders': None, # 特征数据边界（最大最小边界）、会影响缺失值的处理（nan_mode取值Min、Max时），默认值None、在训练时特征取值的最大最小值即为特征值边界
    'boosting_type': 'Plain', # 提升类型，取值Ordered（catboost特有的排序提升，在小数据集上效果可能更好，但是运行速度较慢）、Plain（经典提升）
    'max_ctr_complexity': 4, # 分类特征交叉的最高阶数，默认值4
    'logging_level':'Verbose', # 模型训练过程的信息输出等级，取值Silent（不输出信息）、Verbose（默认值，输出评估指标、已训练时间、剩余时间等）、Info（输出额外信息、树的棵树）、Debug（debug信息）
    'metric_period': 1, # 计算目标值、评估指标的频率，默认值1、即每次迭代都输出目标值、评估指标
    'early_stopping_rounds': 500,
    'border_count': 254, # 数值型特征的分箱数，别名max_bin，取值范围[1,65535]、默认值254（CPU下), # 设置提前停止训练，在得到最佳的评估结果后、再迭代n（参数值为n）次停止训练，默认值不启用
    'feature_border_type': 'GreedyLogSum', # 数值型特征的分箱方法，取值Median、Uniform、UniformAndQuantiles、MaxLogSum、MinEntropy、GreedyLogSum（默认值）
    'random_strength':100,
    
}
f1=[]
auc=[]
acc=[]
sn=[]
sp=[]
p=[]
mcc=[]
prc=[]
for i in range(1,6):

    model=CatBoostClassifier(**params)

    model.load_model('./results/catboostmodel{}'.format(i))

    path='./shunxu/'
    fold=i
    dim=50

    x_test= np.concatenate((np.load('{}/feature{}/deepfeature_test.npy'.format(path,fold)),np.load('{}feature{}/H_test{}.npy'.format(path,fold,dim))), axis=1)
    y_test = np.load('{}feature{}/test_label.npy'.format(path,fold))

    s=[]
    for i in range(x_test.shape[0]):
        pre=model.predict_proba(x_test[i])
        s.append(pre[1])

    F1,AUC,ACC,Sn,Sp,Pre,Mcc,PRC=evalue(s,y_test)
    
    f1.append(F1)
    auc.append(AUC)
    acc.append(ACC)
    sn.append(Sn)
    sp.append(Sp)
    p.append(Pre)
    mcc.append(Mcc)
    prc.append(PRC)
    
print(np.mean(f1),np.mean(auc),np.mean(acc),np.mean(sn),np.mean(sp),np.mean(p),np.mean(mcc),np.mean(prc))
    
    
    
# def evalue():
#     from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,average_precision_score,roc_curve



#     w1=0.6
#     w2=0.2
#     w3=0.2

#     l=np.array(l).astype('float')
#     s=[s1[i][0]*w1+s2[i][0]*w2+s3[i][0]*w3 for i in range(len(s1))]
#     p_new=[]

#     FPR, recall, thresholds1 = roc_curve(l,s, pos_label=1)
#     gmean=np.sqrt(recall*(1-FPR))
#     index=np.argmax(gmean)
#     th1=thresholds1[index]
#     print(th1)


#     for i in s:
#         if i<th1:
#             p_new.append(0)
#         else:
#             p_new.append(1)
#     # s=s3

#     p=p_new

#     tn,fp,fn,tp = confusion_matrix(l, p).ravel() 
#     print(tn,fp,fn,tp)
#     pp = tp+fn
#     pn = tn+fp
#     sensitivity = tp / pp
#     specificity = tn / pn
#     precision = tp / (tp + fp)
#     acc = (tp+tn) / (pp+pn)
#     mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
#     print("F1-Score:{:.4f}".format(f1_score(l,p)))
#     print("AUC:{:.4f}".format(roc_auc_score(l,s)))
#     print("ACC:{:.4f}".format(acc))
#     print("Sn:{:.4f}".format(sensitivity))
#     print("Sp:{:.4f}".format(specificity))
#     print("Pre:{:.4f}".format(precision))
#     print("Mcc:{:.4f}".format(mcc))
#     print("PRC:{:.4f}".format(average_precision_score(l,s)))
    
