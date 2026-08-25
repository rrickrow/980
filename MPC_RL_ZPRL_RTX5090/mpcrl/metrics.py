from __future__ import annotations
import csv, os
import numpy as np


def infer_success(info: dict, episode_return: float, threshold=None):
    for key in ('success','is_success'):
        if key in info: return float(bool(info[key]))
    if threshold is not None: return float(episode_return >= float(threshold))
    return float('nan')


class EpisodeMetrics:
    def __init__(self): self.reset()
    def reset(self):
        self.ret=0.0; self.n=0; self.mpc_ms=[]; self.pred_err=[]; self.unc=[]; self.resnorm=[]; self.gates=[]; self.actions=[]
    def step(self,reward,action,mpc_ms=np.nan,pred_err=np.nan,unc=np.nan,resnorm=np.nan,gate=np.nan):
        self.ret+=float(reward); self.n+=1; self.actions.append(np.asarray(action,np.float32).copy())
        self.mpc_ms.append(mpc_ms); self.pred_err.append(pred_err); self.unc.append(unc); self.resnorm.append(resnorm); self.gates.append(gate)
    def finish(self,info,success_threshold=None):
        a=np.asarray(self.actions)
        d1=np.linalg.norm(np.diff(a,axis=0),axis=1).mean() if len(a)>1 else np.nan
        d2=np.linalg.norm(np.diff(a,n=2,axis=0),axis=1).mean() if len(a)>2 else np.nan
        nm=lambda x: float(np.nanmean(x)) if np.any(np.isfinite(x)) else float('nan')
        return {'episode_return':self.ret,'episode_length':self.n,'success':infer_success(info,self.ret,success_threshold),
                'mpc_ms':nm(np.asarray(self.mpc_ms,float)),'prediction_mse':nm(np.asarray(self.pred_err,float)),
                'uncertainty':nm(np.asarray(self.unc,float)),'residual_norm':nm(np.asarray(self.resnorm,float)),
                'gate':nm(np.asarray(self.gates,float)),'action_d1':float(d1),'action_d2':float(d2)}


class CSVLogger:
    def __init__(self,path,fieldnames):
        os.makedirs(os.path.dirname(path),exist_ok=True); self.path=path; self.fieldnames=fieldnames
        if not os.path.exists(path):
            with open(path,'w',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=fieldnames).writeheader()
    def write(self,row):
        with open(self.path,'a',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=self.fieldnames).writerow({k:row.get(k,'') for k in self.fieldnames})
