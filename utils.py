import torch

def precision(preds,targets):
    return list(map(lambda x,y:len(torch.nonzero(x))/len(torch.nonzero(y)),preds*targets,targets))

def recall(preds,targets):
    return list(map(lambda x,y:len(torch.nonzero(x))/len(torch.nonzero(y)),preds*targets,preds))

def f1(preds,targets):
    pres=precision(preds,targets)
    recs=recall(preds,targets)
    return list(map(lambda x,y:2*x*y/(x+y) if x+y!=0 else 0, pres,recs))

def hit(preds,targets):
    return len(torch.nonzero(preds*targets))