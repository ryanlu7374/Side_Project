import pandas as pd
import os
import time
from itertools import product
from pathlib import Path
import numpy as np
import shutil

"""Create switch struction"""
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
    
class Record:
    def __init__(self, path):
        self.path = path

    def writefile(self, inf_list, inf_type):        
        with open(self.path,'a+') as f:
            for case in switch(inf_type):
                if case('Epoch'):
                    name_list = ["epoch","lr","times","train_loss","train_acc","valid_loss","valid_acc"]
                    f.write("VAR_Epoch: ")
                    for (name,item) in zip(name_list,inf_list):
                        f.write("%s," % (item))
                    f.write('\n')
                elif case('Info'):
                    inf_list.append(time.strftime("%Y-%m-%d %H:%M", time.localtime()))
                    name_list = ["description","time"]
                    f.write("VAR_Info: ")
                    for (name,item) in zip(name_list,inf_list):
                        f.write("%s," % (item))
                    f.write('\n')
                elif case('Error'):
                    inf_list.append(time.strftime("%Y-%m-%d %H:%M", time.localtime()))
                    name_list = ["code","description","time"]
                    f.write("VAR_Error: ")
                    for (name,item) in zip(name_list,inf_list):
                        f.write("%s," % (item))
                    f.write('\n')
                else:
                    print("default condition")
                    
    def epoch_df(self, epoch, lr, times,train_loss,train_acc,valid_loss,valid_acc):
        row_list=[]
        df=pd.DataFrame(columns=["epoch","lr","times","train_loss","train_acc","valid_loss","valid_acc"])
        dic_data = {'epoch':epoch,'lr':lr,'times':times,'train_loss':train_loss,'train_acc':train_acc,'valid_loss':valid_loss,'valid_acc':valid_acc}
        data = pd.Series(dic_data)

        df1 = df.append(data,ignore_index=True)
        df1 = df1.round({'epoch':0,'lr':7,'times':2,'train_loss':5,'train_acc':5,'valid_loss':5,'valid_acc':5})
        df_list=df1.loc[0].tolist()
        Record.writefile(self, inf_list=df_list,inf_type='Epoch')
    
def cb_df(cbepoch,cblr,cbtime,cbtrain_loss,cbtrain_acc,cbvalid_loss,cbvalid_acc):
    df["epoch"]=cbepoch.epoch
    df["lr"]=cblr.lr[:-1]
    df["times"]=cbtimes.times
    df["train_loss"]=cbtrain_loss.train_loss
    df["train_acc"]=cbtrain_acc.train_acc
    df["valid_loss"]=cbvalid_loss.valid_loss
    df["valid_loss"]=cbvalid_acc.valid_acc
    df.index +=1
    df.to_csv(newPath+r'\saveLog\history.csv',sep='\t')
    
def create_folder(rootPath):
    time_ = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    imgclsPath = '\\'.join([rootPath,'saveLog',time_])
    if not os.path.isdir(imgclsPath):
        os.mkdir(imgclsPath)

    #label name permutation with train label
    TrnNamelist = os.listdir(rootPath + r'\train')
    TrnNamePermutation = list(product(os.listdir(rootPath + r'\train'), repeat=2))

    Picfile=[]
    for TstName in os.listdir(rootPath + r'\test'):
        for item in TrnNamePermutation:
            if item[0] == TstName:
                if not os.path.isdir('\\'.join([imgclsPath,item[0]+'_'+item[1]])):
                        os.mkdir('\\'.join([imgclsPath,item[0]+'_'+item[1]]))
                        Picfile.append([item[0],item[1]])
    
    return TrnNamelist, imgclsPath

def output_df(trn_name_list,x_test_id,prediction):
    list_=["file_name","label","pred","score","same","cond","path","NG_score"]
    
    for idx, names in enumerate(trn_name_list):
        list_.append("score_"+names)    
    output_df=pd.DataFrame(columns=list_)
    
    for idx, names in enumerate(trn_name_list):
        output_df["score_"+names]=np.round(prediction[:,idx],4).tolist()
    
    file=[]
    for file_name in x_test_id:
        fn=Path(file_name)
        file.append(fn.name.lower())
    output_df["file_name"]=file
    
    y_test_pred = prediction.argmax(axis=-1).astype(np.int32)
    output_df["pred"]= y_test_pred

    output_df["path"]=x_test_id

    return output_df

def file_move(output_df,trn_name_list,img_cls_path,y_test):    

    output_df["label"]=y_test.astype(np.int32)
    
    for idx, names in enumerate(trn_name_list):
        output_df.label[output_df["label"]==idx]=names
        output_df.pred[output_df["pred"]==idx]=names
        if names=='pass': 
            passidx = idx
        elif names=='reject': 
            rejectidx = idx

    for idx,paths in enumerate(output_df["path"]):
        if(output_df.loc[idx,"label"]=='pass' and output_df.loc[idx,"pred"]=='reject' and 
           output_df.loc[idx,"score"][rejectidx]<=0.8):
            shutil.copyfile(paths,'\\'.join([img_cls_path,output_df.loc[idx,"label"]+'_'+output_df.loc[idx,"label"],output_df.loc[idx,"file_name"]]))
        else:
            shutil.copyfile(paths,'\\'.join([img_cls_path,output_df.loc[idx,"label"]+'_'+output_df.loc[idx,"pred"],output_df.loc[idx,"file_name"]]))