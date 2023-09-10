import torch
import torch.nn as nn
from dataset import LGCNDataset,RecDataset
from model import GCLRec
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import *
import utils
import random
import numpy as np

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)            # 为CPU设置随机种子
torch.cuda.manual_seed_all(2022)       # 为当前GPU设置随机种子

config = {}
config['batch_size'] = 32
config['lgc_latent_dim'] = 32
config['gru_latent_dim'] = 32
config['lightGCN_n_layers'] = 1
config['gru_num_layers'] = 2
config['layer_norm_eps']=1e-5
config['dropout'] = 0.3
config['device'] = 'cuda:2'
config['train_path'] ="./data/beauty/train.txt"
config['val_path'] ="./data/beauty/val.txt"
config['test_path'] ="./data/beauty/test.txt"
config['case_path'] ="./data/beauty/case.txt"
config['run_type']='train'
config['num_aug']=2
config['n_interest']=1
config['epoch']=60
config['lr']=0.01
config['topk']=[5,10,15,20,25,30]
config['val_k']=10
config['val_step']=1
config['patience']=10
config['weight_decay']=10e-5


def collate_fn(data):
    bseq=[]
    bseq_len=[]
    btar=[]
    for i in data:
        bseq.append(i[0])
        bseq_len.append(i[1])
        btar.append(i[2])

    bseq=pad_sequence(bseq,batch_first=True)
    bseq_len=torch.tensor(bseq_len)
    btar=torch.stack(btar,dim=0)
    '''_,idx_sort=torch.sort(bseq_len, dim=0, descending=True)
    bseq=torch.index_select(bseq,0,idx_sort)#[batch_size,max_basket_seq_len]
    bseq_len=torch.index_select(bseq_len,0,idx_sort)#[batch_size]
    btar=torch.index_select(btar,0,idx_sort)#[batch_size,num_item]'''
    return bseq,bseq_len,btar

def recloss(logits,targets):#logits:[batch_size,num_item]  targets:[batch_size,num_item]
    #print("logits:{}".format(logits))
    rec_loss=-torch.sum(targets*torch.log(logits+10e-24))/torch.sum(targets)-torch.sum((1-targets)*torch.log(1-logits+10e-24))/torch.sum(1-targets)
    return rec_loss

def clloss(batch_pos_bseq_emb,batch_neg_bseq_emb,batch_bseq_emb):
    ce=nn.CrossEntropyLoss()
    sim_pp = torch.matmul(batch_pos_bseq_emb, batch_pos_bseq_emb.T) 
    sim_oo = torch.matmul(batch_bseq_emb, batch_bseq_emb.T)
    sim_po = torch.matmul(batch_pos_bseq_emb, batch_bseq_emb.T)
    d = sim_po.shape[-1]
    #print("sim_pp:{}".format(sim_pp))
    '''sim_pp[..., range(d), range(d)] = 0.0
    sim_oo[..., range(d), range(d)] = 0.0
    raw_scores1 = torch.cat([sim_po, sim_pp], dim=-1)
    raw_scores2 = torch.cat([sim_oo, sim_po.transpose(-1, -2)], dim=-1)
    all_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
    labels = torch.arange(2 * d, dtype=torch.long, device=all_scores.device)
    cl_loss1 = ce(all_scores, labels)'''

    sim_nn = torch.matmul(batch_neg_bseq_emb, batch_neg_bseq_emb.T)
    sim_no = torch.matmul(batch_neg_bseq_emb, batch_bseq_emb.T)
    sim_nn[..., range(d), range(d)] = 0.0
    raw_scores1 = torch.cat([sim_no, sim_nn], dim=-1)
    raw_scores2 = torch.cat([sim_oo, sim_no.transpose(-1, -2)], dim=-1)
    all_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
    labels = torch.arange(2 * d, dtype=torch.long, device=all_scores.device)
    cl_loss2 = ce(all_scores, labels)

    return cl_loss2

def cllossnp(batch_pos_bseq_emb,batch_neg_bseq_emb):
    ce=nn.CrossEntropyLoss()
    sim_pp = torch.matmul(batch_pos_bseq_emb, batch_pos_bseq_emb.T) 
    sim_nn = torch.matmul(batch_neg_bseq_emb, batch_neg_bseq_emb.T)
    sim_pn = torch.matmul(batch_pos_bseq_emb, batch_neg_bseq_emb.T)
    d = sim_pn.shape[-1]
    #print("sim_pp:{}".format(sim_pp))
    sim_pp[..., range(d), range(d)] = 0.0
    sim_nn[..., range(d), range(d)] = 0.0
    raw_scores1 = torch.cat([sim_pn, sim_pp], dim=-1)
    raw_scores2 = torch.cat([sim_nn, sim_pn.transpose(-1, -2)], dim=-1)
    all_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
    labels = torch.arange(2 * d, dtype=torch.long, device=all_scores.device)
    cl_loss = ce(all_scores, labels)
    return cl_loss

def logitsloss(batch_bseq_emb,batch_iseq_emb):
    '''ce=nn.CrossEntropyLoss()
    sim_oo = torch.matmul(batch_bseq_emb, batch_bseq_emb.T)
    sim_ii = torch.matmul(batch_iseq_emb, batch_iseq_emb.T)
    sim_io = torch.matmul(batch_iseq_emb, batch_bseq_emb.T)
    d = sim_io.shape[-1]
    sim_oo[..., range(d), range(d)] = 0.0
    sim_ii[..., range(d), range(d)] = 0.0
    
    raw_scores1 = torch.cat([sim_io, sim_ii], dim=-1)
    raw_scores2 = torch.cat([sim_oo, sim_io.transpose(-1, -2)], dim=-1)
    all_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
    labels = torch.arange(2 * d, dtype=torch.long, device=all_scores.device)
    cl_loss3 = ce(all_scores, labels)'''
    cl_loss3=torch.mean(torch.pow(batch_bseq_emb-batch_iseq_emb,2))
    #cl_loss3=-torch.sum(torch.sigmoid(batch_bseq_emb)*torch.log(torch.sigmoid(batch_iseq_emb)))
    return cl_loss3

def clloss2(batch_logits,batch_pos_logits,batch_neg_logits,batch_labels):
    '''print(batch_logits)
    print(batch_pos_logits)
    print(batch_neg_logits)
    print(batch_labels)'''
    loss=0
    loss=loss+torch.sum(torch.sigmoid(batch_neg_logits-batch_logits)*batch_labels)
    loss=loss+torch.sum(torch.sigmoid(batch_logits-batch_pos_logits)*batch_labels)
    return loss

if __name__=="__main__":

    lgcn_dataset = LGCNDataset(config['train_path'], config['val_path'], config['test_path'])

    #图计算basket和item表征
    gclrec=GCLRec(config,lgcn_dataset).to(config['device'])
    if config['run_type']=='train':
        best_val=0
        count=0
        optimizer=torch.optim.Adam(gclrec.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        train_dataset=RecDataset(config['train_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
        train_recloader=DataLoader(train_dataset,batch_size=config['batch_size'],collate_fn=collate_fn,drop_last=True)
        #gclrec.load_state_dict(torch.load('beauty_rec.pth'),strict=False)
        #gclrec.load_state_dict(torch.load('oncl_beauty_i4_gcn1_gru2.pth'))
        print("Begin to train......")
        for e in range(config['epoch']):
            gclrec=gclrec.train()
            gclrec.g_droped = None
            gclrec.pos_g_droped = None
            gclrec.neg_g_droped = None
            gclrec.get_graph(run_type='aug')
            gclrec.augment()
            gclrec.get_graph(run_type='train')
            print("Graphs construct Finish...")
            for batch_id,(bseq,bseq_len,btar) in tqdm(enumerate(train_recloader)):
                bseq=bseq.to(config['device'])
                bseq_len=bseq_len.to(config['device'])
                btar=btar.to(config['device'])
                logits, batch_pos_bseq_emb, batch_neg_bseq_emb, batch_bseq_emb,item_logits,basket_logits=gclrec(bseq, bseq_len,'train')
                #logits,pos_logits,neg_logits=gclrec(bseq, bseq_len,'train')
                '''print("logits.shape:{}".format(logits.shape))
                print("btar:{}".format(btar.shape))
                print("batch_pos_bseq_emb.shape:{}".format(batch_pos_bseq_emb.shape))
                print("batch_neg_bseq_emb.shape:{}".format(batch_neg_bseq_emb.shape))
                print("batch_bseq_emb.shape:{}".format(batch_bseq_emb.shape))'''
                #print("logits[:10,:15]:{}".format(logits[:10,:]))
                rec_loss=recloss(logits,btar)
                #cl_loss=clloss2(logits,pos_logits,neg_logits,btar)
                #cl_loss=clloss(batch_pos_bseq_emb, batch_neg_bseq_emb, batch_bseq_emb)
                cl_loss=cllossnp(batch_pos_bseq_emb, batch_neg_bseq_emb)
                logits_loss=logitsloss(item_logits,basket_logits)
                print("======Train loss======")
                print("rec_loss:{}".format(rec_loss))
                print("cl_loss:{}".format(cl_loss))
                print("logits_loss:{}".format(logits_loss))

                #loss=rec_loss+cl_loss*config['cl_weight']+logits_loss*config['emb_weight']
                loss=rec_loss+cl_loss+logits_loss
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(gclrec.parameters(), max_norm=1, norm_type=2)
                optimizer.step()
            
            if e%config['val_step']==0:
                gclrec.eval()
                gclrec.get_graph(run_type='val')
                val_dataset=RecDataset(config['val_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
                val_recloader=DataLoader(val_dataset,batch_size=config['batch_size'],collate_fn=collate_fn,drop_last=True)
                f1s={}
                hits={}
                gts=0
                sample_num=0
                for k in config['topk']:
                    f1s[k]=0
                    hits[k]=0
                for batch_id,(bseq,bseq_len,btar) in tqdm(enumerate(val_recloader)):
                    sample_num=sample_num+len(bseq_len)
                    gts=gts+len(torch.nonzero(btar))
                    bseq=bseq.to(config['device'])
                    bseq_len=bseq_len.to(config['device'])
                    btar=btar.to(config['device'])
                    logits=gclrec(bseq, bseq_len,'val')  #[batch_size,num_items]
                    for k in config['topk']:
                        preds=torch.topk(logits,dim=1,k=k).indices   #[batch_size,k]
                        preds=torch.zeros_like(logits).scatter_(1,preds,1)
                        f1s[k]=f1s[k]+sum(utils.f1(preds,btar))
                        hits[k]=hits[k]+utils.hit(preds,btar)
                del val_dataset,val_recloader
                with open('beauty_i1_gcn1_gru2_aug2.txt','a') as f:
                    for k in config['topk']:
                        res="Epoch{} Val K={}  f1:{}  hr:{}\n".format(e,k,f1s[k]/sample_num,hits[k]/gts)
                        f.write(res)
                    f.write("\n")
                
                if f1s[config['val_k']]>best_val:
                    best_val=f1s[config['val_k']]
                    count=0
                    torch.save(gclrec.state_dict(), 'beauty_i1_gcn1_gru2_aug2.pth')
                    print("Model of epoch {} is saved.".format(e))
                else:
                    count=count+1
                    print("Counter {} of {}".format(count,config['patience']))
                    if count>=config['patience']:
                        break
            
    elif config['run_type']=='nocl':
        best_val=0
        optimizer=torch.optim.Adam(gclrec.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        train_dataset=RecDataset(config['train_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
        train_recloader=DataLoader(train_dataset,batch_size=config['batch_size'],collate_fn=collate_fn,drop_last=True)
        print("Begin to train......")
        for e in range(config['epoch']):
            gclrec=gclrec.train()
            gclrec.g_droped = None
            gclrec.get_graph(run_type='oncl')
            print("Graphs construct Finish...")
            for batch_id,(bseq,bseq_len,btar) in tqdm(enumerate(train_recloader)):
                bseq=bseq.to(config['device'])
                bseq_len=bseq_len.to(config['device'])
                btar=btar.to(config['device'])
                logits,item_logits,basket_logits=gclrec(bseq, bseq_len,'oncl')
                rec_loss=recloss(logits,btar)
                logits_loss=logitsloss(item_logits,basket_logits)
                loss=rec_loss+logits_loss
                print("======Train loss======")
                print("rec_loss:{}".format(rec_loss))
                print("logits_loss:{}".format(logits_loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if e%config['val_step']==0:
                gclrec.eval()
                gclrec.get_graph(run_type='val')
                val_dataset=RecDataset(config['val_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
                val_recloader=DataLoader(val_dataset,batch_size=config['batch_size'],collate_fn=collate_fn,drop_last=True)
                f1s={}
                hits={}
                gts=0
                sample_num=0
                for k in config['topk']:
                    f1s[k]=0
                    hits[k]=0
                for batch_id,(bseq,bseq_len,btar) in tqdm(enumerate(val_recloader)):
                    sample_num=sample_num+len(bseq_len)
                    gts=gts+len(torch.nonzero(btar))
                    bseq=bseq.to(config['device'])
                    bseq_len=bseq_len.to(config['device'])
                    btar=btar.to(config['device'])
                    logits=gclrec(bseq, bseq_len,'val')  #[batch_size,num_items]
                    for k in config['topk']:
                        preds=torch.topk(logits,dim=1,k=k).indices   #[batch_size,k]
                        preds=torch.zeros_like(logits).scatter_(1,preds,1)
                        f1s[k]=f1s[k]+sum(utils.f1(preds,btar))
                        hits[k]=hits[k]+utils.hit(preds,btar)
                del val_dataset,val_recloader
                with open('beauty_i1_gcn1_gru2_aug2.txt','a') as f:
                    for k in config['topk']:
                        res="Epoch{} Val K={}  f1:{}  hr:{}\n".format(e,k,f1s[k]/sample_num,hits[k]/gts)
                        f.write(res)
                    f.write("\n")
                
                if f1s[config['val_k']]>best_val:
                    best_val=f1s[config['val_k']]
                    count=0
                    torch.save(gclrec.state_dict(), 'beauty_i1_gcn1_gru2_aug2.pth')
                    print("Model of epoch {} is saved.".format(e))
                else:
                    count=count+1
                    print("Counter {} of {}".format(count,config['patience']))
                    if count>=config['patience']:
                        break

    elif config['run_type']=='test':
        gclrec.eval()
        gclrec.get_graph(run_type='test')
        test_dataset=RecDataset(config['test_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
        test_recloader=DataLoader(test_dataset,batch_size=config['batch_size'],collate_fn=collate_fn,drop_last=True)
        gclrec.load_state_dict(torch.load('beauty_i1_gcn1_gru2_aug2.pth'))
        f1s={}
        hits={}
        gts=0
        sample_num=0
        for k in config['topk']:
            f1s[k]=0
            hits[k]=0
        for batch_id,(bseq,bseq_len,btar) in tqdm(enumerate(test_recloader)):
            sample_num=sample_num+len(bseq_len)
            gts=gts+len(torch.nonzero(btar))
            bseq=bseq.to(config['device'])
            bseq_len=bseq_len.to(config['device'])
            btar=btar.to(config['device'])
            logits=gclrec(bseq, bseq_len,'test')  #[batch_size,num_items]
            for k in config['topk']:
                preds=torch.topk(logits,dim=1,k=k).indices   #[batch_size,k]
                preds=torch.zeros_like(logits).scatter_(1,preds,1)
                f1s[k]=f1s[k]+sum(utils.f1(preds,btar))
                hits[k]=hits[k]+utils.hit(preds,btar)
        
        with open('beauty_i1_gcn1_gru2_aug2.txt','a') as f:
            for k in config['topk']:
                res="Test K={}  f1:{}  hr:{}\n".format(k,f1s[k]/sample_num,hits[k]/gts)
                f.write(res)
            f.write("\n")


    elif config['run_type']=='case':
        gclrec.eval()
        gclrec.get_graph(run_type='test')
        case_dataset=RecDataset(config['case_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
        case_recloader=DataLoader(case_dataset,batch_size=config['batch_size'],collate_fn=collate_fn,drop_last=True)
        gclrec.load_state_dict(torch.load('beauty_i4_gcn2_gru2.pth'))
        for batch_id,(bseq,bseq_len,btar) in tqdm(enumerate(case_recloader)):
            bseq=bseq.to(config['device'])
            bseq_len=bseq_len.to(config['device'])
            btar=btar.to(config['device'])
            logits=gclrec(bseq, bseq_len,'test')  #[batch_size,num_items]
            print("Case Study Finish!")
