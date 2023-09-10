"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
from asyncio import base_tasks
from ctypes import sizeof
from operator import index
import torch
from dataset import LGCNDataset
from torch import index_select, log, nn, transpose
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence
import numpy as np
import time
import pandas as pd
import pickle


class GCLRec(nn.Module):
    def __init__(self, 
                 config:dict, 
                 dataset:LGCNDataset):
        super(GCLRec, self).__init__()
        self.config = config
        self.dataset = dataset
        self.num_baskets  = self.dataset.num_baskets
        self.num_items  = self.dataset.num_items
        self.id2basket_dict=self.dataset.id2basket_dict
        self.n_layers = self.config['lightGCN_n_layers']
        self.n_interest=self.config['n_interest']

        self.emb_dropout = nn.Dropout(config['dropout'])
        self.gru=nn.GRU(input_size=config['lgc_latent_dim'], 
                        hidden_size=config['gru_latent_dim'], 
                        num_layers=config['gru_num_layers'],
                        dropout=config['dropout'],
                        batch_first=True)
        self.bn1=nn.BatchNorm1d(self.config['gru_latent_dim'])
        self.linear=nn.Linear(config['gru_latent_dim'],config['lgc_latent_dim'],bias=True)
        self.bn2=nn.BatchNorm1d(config['lgc_latent_dim'])
        self.ln = nn.LayerNorm(config['lgc_latent_dim'], eps=config['layer_norm_eps'])
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax(dim=2)
        self.relu=nn.ReLU()
        self.item_gru = nn.GRU(
            input_size=self.config['lgc_latent_dim'],
            hidden_size=config['gru_latent_dim'],
            num_layers=config['gru_num_layers'],
            dropout=config['dropout'],
            batch_first=True,
        )
        self.item_linear=nn.Linear(config['gru_latent_dim'],config['lgc_latent_dim'],bias=True)
        #self.item_hidden_state=self.hidden_state=torch.randn(self.config['gru_num_layers'], self.config['batch_size'], self.config['gru_latent_dim']).to(self.config['device'])
        self.linear_bseq_interest=nn.Linear(config['lgc_latent_dim'],self.n_interest*config['lgc_latent_dim'],bias=False)
        self.linear_iseq_interest=nn.Linear(config['lgc_latent_dim'],self.n_interest*config['lgc_latent_dim'],bias=False)
        self.merge_bakset_interest=nn.Linear(self.n_interest,1,bias=False)
        self.merge_item_interest=nn.Linear(self.n_interest,1,bias=False)


        self.embedding_basket = torch.nn.Embedding(num_embeddings=self.num_baskets, embedding_dim=self.config['lgc_latent_dim'], padding_idx=0)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.config['lgc_latent_dim'], padding_idx=0)
        #self.hidden_state=torch.randn(self.config['gru_num_layers'], self.config['batch_size'], self.config['gru_latent_dim']).to(self.config['device'])
        nn.init.normal_(self.embedding_basket.weight, mean=0, std=0.1)
        nn.init.normal_(self.embedding_item.weight, mean=0, std=0.1)

        self.lambda1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lambda2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        #self.all_linear=nn.Linear(2*config['gru_latent_dim'],config['lgc_latent_dim'],bias=True)
        

    def get_graph(self, run_type='train'):
        if run_type=='train':
            self.g_droped = self.dataset.getSparseGraph(graph_type='original',run_type='train') 
            self.pos_g_droped = self.dataset.getSparseGraph(graph_type='pos',run_type='train') 
            self.neg_g_droped = self.dataset.getSparseGraph(graph_type='neg',run_type='train') 
        elif run_type=='val':
            self.g_droped = self.dataset.getSparseGraph(graph_type='original',run_type='val') 
        elif run_type=='test':
            self.g_droped = self.dataset.getSparseGraph(graph_type='original',run_type='test') 
        elif run_type=='aug':
            self.g_droped = self.dataset.getSparseGraph(graph_type='original',run_type='train')
        elif run_type=='nocl':
            self.g_droped = self.dataset.getSparseGraph(graph_type='original',run_type='train') 
     
    def computer(self, graph_type='original'):
        """
        propagate methods for lightGCN
        """       
        #start2=time()
        baskets_emb = self.embedding_basket.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([baskets_emb, items_emb])
        embs = [all_emb]
        #print("pre time:{}".format(time()-start2))
           
        #print("getSparseGraph time:{}".format(time()-start2))
        '''g_droped = self.dataset.getSparseGraph(graph_type,run_type) 
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)'''
        #print("mm time:{}".format(time()-start2))
        
        if graph_type=='original':
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(self.g_droped, all_emb)
                embs.append(all_emb)
        elif graph_type=='pos':
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(self.pos_g_droped, all_emb)
                embs.append(all_emb)
        elif graph_type=='neg':
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(self.neg_g_droped, all_emb)
                embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        baskets, items = torch.split(light_out, [self.num_baskets, self.num_items])
        if graph_type=='original':
            return baskets.to(self.config['device']), items.to(self.config['device'])
        else:
            return baskets
    
    def augment(self):
        baskets, items=self.computer(graph_type='original')    #num_baskets*latent_dim, num_item*latent_dim
        baskets=baskets.detach().cpu()
        items=items.detach().cpu()
        #图增广
        basket_item_intensity=torch.mm(baskets,items.t()) #num_baskets*num_item
        item_filter=torch.argsort(basket_item_intensity,dim=1,descending=True) #num_baskets*num_item
        del basket_item_intensity
        aug_trainBasket=np.array([])
        aug_train_b2i_weight=np.array([])
        pos_trainItem=np.array([])
        neg_trainItem=np.array([])
        for i in range(1,len(item_filter)+1):
            if i in self.dataset.trainBasket:
                aug_trainBasket=np.append(aug_trainBasket, [i]*self.config['num_aug'])
                aug_train_b2i_weight=np.append(aug_train_b2i_weight,[1]*self.config['num_aug'])
                pos_trainItem=np.append(pos_trainItem,item_filter[i][:self.config['num_aug']])
                neg_trainItem=np.append(neg_trainItem,item_filter[i][-self.config['num_aug']:])
                '''print("item_filter[i][:self.config['num_aug']]:{}".format(item_filter[i][:self.config['num_aug']]))
                print("item_filter[i][-self.config['num_aug']:]:{}".format(item_filter[i][-self.config['num_aug']:]))'''
        self.dataset.aug_trainBasket=np.append(self.dataset.trainBasket, aug_trainBasket)
        self.dataset.aug_train_b2i_weight=np.append(self.dataset.train_b2i_weight,aug_train_b2i_weight)
        self.dataset.pos_trainItem=np.append(self.dataset.trainItem, pos_trainItem)
        self.dataset.neg_trainItem=np.append(self.dataset.trainItem, neg_trainItem)
        del aug_trainBasket,aug_train_b2i_weight,pos_trainItem,neg_trainItem,item_filter
        print("Data augmentation Finish...")


    def gather_embs(self, embs, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        embs = embs.reshape(1,embs.shape[0],embs.shape[1]).expand(gather_index.shape[0],-1,-1)
        gather_index = gather_index.reshape(gather_index.shape[0], gather_index.shape[1], 1).expand(-1, -1, embs.shape[-1])
        embs_tensor = embs.gather(dim=1, index=gather_index)
        return embs_tensor
    
    def gather_indexes(self, output, gather_index):#[4,batch_size,max_basket_seq_len,latent_dim],[batch_size,]
        #print("output:{}".format(output.shape))
        #print("gather_index1:{}".format(gather_index.shape))
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.reshape(1,-1, 1, 1).expand(output.shape[0],-1, -1, output.shape[-1])
        #print("gather_index2:{}".format(gather_index.shape))
        output_tensor = output.gather(dim=2, index=gather_index)
        return output_tensor.squeeze(2)
    

    def forward(self, bseq, bseq_len, run_type='train'):
        #start1=time()
        #print("ori computer time:{}".format(time()-start1))
        if run_type=='train':
            baskets, items=self.computer(graph_type='original')    #num_baskets*latent_dim, num_item*latent_dim
            pos_baskets=self.computer(graph_type='pos')    #num_baskets*latent_dim, num_item*latent_dim
            neg_baskets=self.computer(graph_type='neg')    #num_baskets*latent_dim, num_item*latent_dim

            '''print("baskets:{}".format(baskets[5:7,:15]))
            print("items:{}".format(items[5:7,:15]))
            print("self.embedding_basket.weight:{}".format(self.embedding_basket.weight[5:7,:15]))
            print("self.embedding_item.weight:{}".format(self.embedding_item.weight[5:7,:15]))'''
        else:
            baskets, items=self.computer(graph_type='original')    #num_baskets*latent_dim, num_item*latent_dim

        
            

        '''with pd.ExcelWriter("MergedResults.xlsx") as writer:
            all_data=pd.DataFrame(baskets[:100],dtype=float)
            all_data.to_excel(writer,sheet_name="baskets")  
            all_data=pd.DataFrame(items[:100],dtype=float)
            all_data.to_excel(writer,sheet_name="items")  
            all_data=pd.DataFrame(pos_baskets[:100],dtype=float)
            all_data.to_excel(writer,sheet_name="pos_baskets")  
            all_data=pd.DataFrame(bseq[:100],dtype=float)
            all_data.to_excel(writer,sheet_name="bseq")  '''


        #原图序列表征
        #basket_bseq_emb=[]
        
        item_seq=[]
        item_seq_len=[]
        for b in bseq:
            #basket_bseq_emb.append(torch.index_select(baskets,0,b)) #[batch_size,max_basket_seq_len,latent_dim]
            item=[]
            for bid in b:
                if int(bid)!=0:
                    item=item+self.id2basket_dict[int(bid)]
            item_seq.append(torch.as_tensor(item))
            item_seq_len.append(len(item))
        item_seq=pad_sequence(item_seq,batch_first=True).to(self.config['device'])#[batch_size,max_item_seq_len]
        item_seq_len=torch.as_tensor(item_seq_len).to(self.config['device'])
        #basket_bseq_emb=torch.stack(basket_bseq_emb,dim=0).to(self.config['device'])

        '''print("basket_bseq_emb.grad_fn:{}".format(basket_bseq_emb.grad_fn))
        print("basket_bseq_emb.requires_grad:{}".format(basket_bseq_emb.requires_grad))'''
        #print("basket_sqe_emb.shape:{}".format(basket_bseq_emb.shape))
        basket_bseq_emb=self.gather_embs(baskets,bseq)#[batch_size,max_basket_seq_len,latent_dim]
        basket_bseq_emb=self.linear_bseq_interest(basket_bseq_emb)#[batch_size,max_basket_seq_len,latent_dim*4]
        basket_bseq_emb=basket_bseq_emb.reshape(basket_bseq_emb.shape[0],basket_bseq_emb.shape[1],-1,self.n_interest)#[batch_size,max_basket_seq_len,latent_dim,4]
        basket_bseq_emb=basket_bseq_emb.permute(3,0,1,2)#[4,batch_size,max_basket_seq_len,latent_dim]
        basket_bseq_emb=basket_bseq_emb.reshape(-1,basket_bseq_emb.shape[2],basket_bseq_emb.shape[3])#[4*batch_size,max_basket_seq_len,latent_dim]
        basket_bseq_repre,_=self.gru(basket_bseq_emb) #[4*batch_size,max_basket_seq_len,latent_dim]
        basket_bseq_repre=basket_bseq_repre.reshape(self.n_interest,-1,basket_bseq_repre.shape[1],basket_bseq_repre.shape[2])#[4,batch_size,max_basket_seq_len,latent_dim]
        batch_bseq_emb=self.gather_indexes(basket_bseq_repre,bseq_len-1)#[4,batch_size,latent_dim]
        batch_bseq_emb=batch_bseq_emb.permute(1,0,2)#[batch_size,4,latent_dim]
        interest_atten=torch.matmul(batch_bseq_emb,batch_bseq_emb.permute(0,2,1))#[batch_size,4,4]
        interest_atten=self.softmax(interest_atten)
        with open('basket_atten.pickle','wb') as f:
            pickle.dump(interest_atten,f)
        basket_logits=torch.matmul(batch_bseq_emb,items.T)#[batch_size,4,num_item]
        basket_logits=basket_logits.permute(0,2,1)#[batch_size,num_item,4]
        basket_logits=torch.matmul(basket_logits,interest_atten)#[batch_size,num_item,4]
        basket_logits=self.merge_bakset_interest(basket_logits)#[batch_size,num_item,1]
        basket_logits=basket_logits.squeeze(2)



        #basket_bseq_emb=self.emb_dropout(basket_bseq_emb)
        #batch_basket_seq=pack_padded_sequence(basket_bseq_emb, bseq_len.cpu(), batch_first=True)

        '''basket_bseq_repre,_=self.gru(basket_bseq_emb)   
        #print("gru time:{}".format(time()-start1))
        #basket_bseq_repre,_=nn.utils.rnn.pad_packed_sequence(basket_bseq_repre, batch_first=True)#basket_repre:[bs,bakset_seq_len,basket_emb]
        #print("basket_bseq_repre.shape:{}".format(basket_bseq_repre.shape))
        #print("pad time:{}".format(time()-start1))
        batch_bseq_emb=self.gather_indexes(basket_bseq_repre,bseq_len-1)
        basket_logits=torch.mm(batch_bseq_emb,self.embedding_item.weight.T)'''


        '''batch_bseq_emb=[]
        for i in range(len(bseq_len)):
            batch_bseq_emb.append(torch.index_select(basket_bseq_repre[i],0,bseq_len[i]-1)[0])
        batch_bseq_emb=torch.stack(batch_bseq_emb,dim=0).to(self.config['device'])'''
        '''basket_logits=self.linear(batch_bseq_emb)
        basket_logits=torch.mm(basket_logits,items.T)'''
        #basket_logits=torch.mm(batch_bseq_emb,items.T)
        #basket_logits=self.sigmoid(basket_logits)
        #print("basket_logits:{}".format(basket_logits))



        '''_,idx_sort=torch.sort(torch.tensor(item_seq_len), dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        item_seq=torch.index_select(item_seq,0,idx_sort).to(self.config['device'])
        item_seq_len=torch.index_select(torch.tensor(item_seq_len),0,idx_sort)'''
        '''item_seq_emb=[]
        for i in item_seq:
            item_seq_emb.append(torch.index_select(items,0,i))
        item_seq_emb=torch.stack(item_seq_emb,dim=0)'''
        #print("item_seq_emb.shape:{}".format(item_seq_emb.shape))
        #item_seq_emb=pack_padded_sequence(item_seq_emb,item_seq_len,batch_first=True,enforce_sorted=True)
        '''item_seq_emb=self.gather_embs(items,item_seq)
        item_seq_repre,_=self.item_gru(item_seq_emb.to(self.config['device']))
        #item_seq_repre,_=nn.utils.rnn.pad_packed_sequence(item_seq_repre, batch_first=True)#item_seq_repre:[bs,item_seq_len,basket_emb]
        batch_iseq_emb=self.gather_indexes(item_seq_repre,item_seq_len-1)'''
        
        item_seq_emb=self.gather_embs(items,item_seq)#[batch_size,max_item_seq_len,latent_dim]
        item_seq_emb=self.linear_iseq_interest(item_seq_emb)#[batch_size,max_item_seq_len,latent_dim*4]
        item_seq_emb=item_seq_emb.reshape(item_seq_emb.shape[0],item_seq_emb.shape[1],-1,self.n_interest)#[batch_size,max_item_seq_len,latent_dim,4]
        item_seq_emb=item_seq_emb.permute(3,0,1,2)#[4,batch_size,max_item_seq_len,latent_dim]
        item_seq_emb=item_seq_emb.reshape(-1,item_seq_emb.shape[2],item_seq_emb.shape[3])#[4*batch_size,max_item_seq_len,latent_dim]
        item_iseq_repre,_=self.item_gru(item_seq_emb) #[4*batch_size,max_item_seq_len,latent_dim]
        item_iseq_repre=item_iseq_repre.reshape(self.n_interest,-1,item_iseq_repre.shape[1],item_iseq_repre.shape[2])#[4,batch_size,max_item_seq_len,latent_dim]
        batch_iseq_emb=self.gather_indexes(item_iseq_repre,item_seq_len-1)#[4,batch_size,latent_dim]
        batch_iseq_emb=batch_iseq_emb.permute(1,0,2)#[batch_size,4,latent_dim]
        interest_atten=torch.matmul(batch_iseq_emb,batch_iseq_emb.permute(0,2,1))#[batch_size,4,4]
        interest_atten=self.softmax(interest_atten)
        with open('items_atten.pickle','wb') as f:
            pickle.dump(interest_atten,f)
        item_logits=torch.matmul(batch_iseq_emb,items.T)#[batch_size,4,num_item]
        item_logits=item_logits.permute(0,2,1)#[batch_size,num_item,4]
        item_logits=torch.matmul(item_logits,interest_atten)#[batch_size,num_item,4]
        item_logits=self.merge_item_interest(item_logits)#[batch_size,num_item,1]
        item_logits=item_logits.squeeze(2)
        

        '''batch_iseq_emb=[]
        for i in range(len(item_seq_len)):
            batch_iseq_emb.append(torch.index_select(item_seq_repre[i].cpu(),0,item_seq_len[i]-1)[0])
        batch_iseq_emb=torch.stack(batch_iseq_emb,dim=0).to(self.config['device'])'''
        #print("batch_iseq_emb.shape:{}".format(batch_iseq_emb.shape))
        #batch_iseq_emb=torch.index_select(batch_iseq_emb,0,idx_unsort.to(self.config['device']))
        '''item_logits=self.item_linear(batch_iseq_emb)
        item_logits=torch.mm(item_logits,items.T)'''
        #item_logits=torch.mm(batch_iseq_emb,self.embedding_item.weight.T)
        #item_logits=torch.mm(batch_iseq_emb,items.T)
        #item_logits=self.sigmoid(item_logits)
        #print("item_logits:{}".format(item_logits))

        


        '''with pd.ExcelWriter("MergedResults.xlsx") as writer:
            all_data=pd.DataFrame(list(basket_logits.cpu().detach())[:100],dtype=float)
            all_data.to_excel(writer,sheet_name="basket_logits")  
            all_data=pd.DataFrame(list(item_logits.cpu().detach())[:100],dtype=float)
            all_data.to_excel(writer,sheet_name="item_logits")  
        print("sleeping..........")
        time.sleep(10)'''
        #print("Prediction Finish...")

        '''logits=torch.cat([batch_bseq_emb,batch_iseq_emb],dim=1)
        logits=self.all_linear(logits)
        logits=torch.mm(logits,items.T)
        logits=self.sigmoid(logits)'''


        logits=basket_logits+item_logits
        logits=self.sigmoid(logits)
        '''print("basket_logits:{}".format(list(basket_logits[:6,10:20])))
        print("item_logits:{}".format(list(item_logits[:6,10:20])))
        print("logits:{}".format(list(logits[:6,10:20])))'''
        

        if run_type=='trai':
            #正图序列表征
            '''basket_pos_bseq_emb=[]
            for b in bseq:
                basket_pos_bseq_emb.append(torch.index_select(pos_baskets,0,b)) #[batch_size,max_basket_seq_len,latent_dim]
            basket_pos_bseq_emb=torch.stack(basket_pos_bseq_emb,dim=0).to(self.config['device'])'''
            basket_pos_bseq_emb=self.gather_embs(pos_baskets,bseq)
            #basket_pos_bseq_emb=self.emb_dropout(basket_pos_bseq_emb)
            #basket_pos_bseq_emb=pack_padded_sequence(basket_pos_bseq_emb, bseq_len.cpu(), batch_first=True)
            basket_pos_bseq_repre,_=self.gru(basket_pos_bseq_emb)  
            #basket_pos_bseq_repre,_=nn.utils.rnn.pad_packed_sequence(basket_pos_bseq_repre, batch_first=True)#basket_repre:[bs,bakset_seq_len,basket_emb]
            batch_pos_bseq_emb=self.gather_indexes(basket_pos_bseq_repre,bseq_len-1)
            '''batch_pos_bseq_emb=[]
            for i in range(len(bseq_len)):
                batch_pos_bseq_emb.append(torch.index_select(basket_pos_bseq_repre[i],0,bseq_len[i]-1)[0])
            batch_pos_bseq_emb=torch.stack(batch_pos_bseq_emb,dim=0).to(self.config['device'])'''
            #print("Positive sequence embed Finish...")
            '''pos_logits=self.linear(batch_pos_bseq_emb)
            pos_logits=torch.mm(pos_logits,items.T)
            pos_logits=self.sigmoid(pos_logits)'''

            #负图序列表征
            '''basket_neg_bseq_emb=[]
            for b in bseq:
                basket_neg_bseq_emb.append(torch.index_select(neg_baskets,0,b)) #[batch_size,max_basket_seq_len,latent_dim]
            basket_neg_bseq_emb=torch.stack(basket_neg_bseq_emb,dim=0).to(self.config['device'])'''
            #basket_neg_bseq_emb=pack_padded_sequence(basket_neg_bseq_emb, bseq_len.cpu(), batch_first=True)
            basket_neg_bseq_emb=self.gather_embs(neg_baskets,bseq)
            #basket_neg_bseq_emb=self.emb_dropout(basket_neg_bseq_emb)
            basket_neg_bseq_repre,_=self.gru(basket_neg_bseq_emb)  
            #basket_neg_bseq_repre,_=nn.utils.rnn.pad_packed_sequence(basket_neg_bseq_repre, batch_first=True)#basket_repre:[bs,bakset_seq_len,basket_emb]
            batch_neg_bseq_emb=self.gather_indexes(basket_neg_bseq_repre,bseq_len-1)
            
            '''batch_neg_bseq_emb=[]
            for i in range(len(bseq_len)):
                batch_neg_bseq_emb.append(torch.index_select(basket_neg_bseq_repre[i],0,bseq_len[i]-1)[0])
            batch_neg_bseq_emb=torch.stack(batch_neg_bseq_emb,dim=0).to(self.config['device'])'''
            #print("Negative sequence embed Finish...")
            

            '''with pd.ExcelWriter("MergedResults3.xlsx") as writer:
                all_data=pd.DataFrame(list(batch_neg_bseq_emb)[:100],dtype=float)
                all_data.to_excel(writer,sheet_name="batch_neg_bseq_emb")  
                all_data=pd.DataFrame(list(batch_pos_bseq_emb)[:100],dtype=float)
                all_data.to_excel(writer,sheet_name="batch_pos_bseq_emb")  

            print("sleeping..........")
            time.sleep(10)'''
            return logits,batch_pos_bseq_emb,batch_neg_bseq_emb,batch_bseq_emb,item_logits,basket_logits
        
        elif run_type=='train':
            basket_bseq_pos_emb=self.gather_embs(pos_baskets,bseq)#[batch_size,max_basket_seq_len,latent_dim]
            basket_bseq_pos_emb=self.linear_bseq_interest(basket_bseq_pos_emb)#[batch_size,max_basket_seq_len,latent_dim*4]
            basket_bseq_pos_emb=basket_bseq_pos_emb.reshape(basket_bseq_pos_emb.shape[0],basket_bseq_pos_emb.shape[1],-1,self.n_interest)#[batch_size,max_basket_seq_len,latent_dim,4]
            basket_bseq_pos_emb=basket_bseq_pos_emb.permute(3,0,1,2)#[4,batch_size,max_basket_seq_len,latent_dim]
            basket_bseq_pos_emb=basket_bseq_pos_emb.reshape(-1,basket_bseq_pos_emb.shape[2],basket_bseq_pos_emb.shape[3])#[4*batch_size,max_basket_seq_len,latent_dim]
            basket_bseq_pos_repre,_=self.gru(basket_bseq_pos_emb) #[4*batch_size,max_basket_seq_len,latent_dim]
            basket_bseq_pos_repre=basket_bseq_pos_repre.reshape(self.n_interest,-1,basket_bseq_pos_repre.shape[1],basket_bseq_pos_repre.shape[2])#[4,batch_size,max_basket_seq_len,latent_dim]
            batch_pos_bseq_emb=self.gather_indexes(basket_bseq_pos_repre,bseq_len-1)#[4,batch_size,latent_dim]
            batch_pos_bseq_emb=batch_pos_bseq_emb.permute(1,2,0)#[batch_size,latent_dim,4]
            batch_pos_bseq_emb=batch_pos_bseq_emb.reshape(batch_pos_bseq_emb.shape[0],-1)#[batch_size,latent_dim*4]

            basket_bseq_neg_emb=self.gather_embs(neg_baskets,bseq)#[batch_size,max_basket_seq_len,latent_dim]
            basket_bseq_neg_emb=self.linear_bseq_interest(basket_bseq_neg_emb)#[batch_size,max_basket_seq_len,latent_dim*4]
            basket_bseq_neg_emb=basket_bseq_neg_emb.reshape(basket_bseq_neg_emb.shape[0],basket_bseq_neg_emb.shape[1],-1,self.n_interest)#[batch_size,max_basket_seq_len,latent_dim,4]
            basket_bseq_neg_emb=basket_bseq_neg_emb.permute(3,0,1,2)#[4,batch_size,max_basket_seq_len,latent_dim]
            basket_bseq_neg_emb=basket_bseq_neg_emb.reshape(-1,basket_bseq_neg_emb.shape[2],basket_bseq_neg_emb.shape[3])#[4*batch_size,max_basket_seq_len,latent_dim]
            basket_bseq_neg_repre,_=self.gru(basket_bseq_neg_emb) #[4*batch_size,max_basket_seq_len,latent_dim]
            basket_bseq_neg_repre=basket_bseq_neg_repre.reshape(self.n_interest,-1,basket_bseq_neg_repre.shape[1],basket_bseq_neg_repre.shape[2])#[4,batch_size,max_basket_seq_len,latent_dim]
            batch_neg_bseq_emb=self.gather_indexes(basket_bseq_neg_repre,bseq_len-1)#[4,batch_size,latent_dim]
            batch_neg_bseq_emb=batch_neg_bseq_emb.permute(1,2,0)#[batch_size,latent_dim,4]
            batch_neg_bseq_emb=batch_neg_bseq_emb.reshape(batch_neg_bseq_emb.shape[0],-1)#[batch_size,latent_dim*4]

            batch_bseq_emb=batch_bseq_emb.permute(0,2,1)
            batch_bseq_emb=batch_bseq_emb.reshape(batch_bseq_emb.shape[0],-1)
            return logits,batch_pos_bseq_emb,batch_neg_bseq_emb,batch_bseq_emb,item_logits,basket_logits
        
        elif run_type=='nocl':
            return logits,item_logits,basket_logits
        
        else:
            return logits
            '''return logits,batch_pos_bseq_emb,batch_neg_bseq_emb,batch_bseq_emb
        
        else:
            return logits'''