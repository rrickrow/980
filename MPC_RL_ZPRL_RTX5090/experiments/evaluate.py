from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np, torch
from mpcrl.config import load_yaml
from mpcrl.envs import make_mujoco_env,dims,action_bounds
from mpcrl.world_model import EnsembleWorldModel
from mpcrl.cem import CEMPlanner
from mpcrl.sac import ResidualSAC
from mpcrl.gate import adaptive_uncertainty_gate
from mpcrl.utils import pick_device,set_seed


def plan_chunk(actions,k):
    p=actions[:k]
    if len(p)<k: p=np.concatenate([p,np.repeat(p[-1:],k-len(p),axis=0)],0)
    return p


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--episodes',type=int,default=10); args=ap.parse_args(); run=Path(args.run_dir); cfg=load_yaml(run/'config.yaml'); device=pick_device(); seed=cfg['env']['seed']; set_seed(seed)
    env,obs,_=make_mujoco_env(cfg['env']['id'],seed+1000); od,ad=dims(env); low,high=action_bounds(env); span=(high-low)/2; w=cfg['world_model']; m=cfg['mpc']; rc=cfg['residual']; sc=cfg['sac']
    ck=torch.load(run/'final.pt',map_location=device); method=ck['method']; wm=EnsembleWorldModel(od,ad,w['ensemble_size'],w['hidden_dim'],w['lr'],w['weight_decay'],device); wm.load_state_dict(ck['world_model']); wm.eval(); planner=CEMPlanner(wm,low,high,m['horizon'],m['candidates'],m['elites'],m['iterations'],m['alpha'],m['init_std'],m['min_std'],m['discount'],w.get('uncertainty_penalty',0.0))
    k=1 if method=='action_residual' else int(rc['chunk_len']); agent=None
    if method!='mpc_only':
        agent=ResidualSAC(od+k*ad,k*ad,sc['hidden_dim'],sc['lr'],sc['gamma'],sc['tau'],sc['init_alpha'],sc['target_entropy_scale'],rc.get('consistency_coef',0.0) if method=='planning_residual' else 0.0,device); agent.load_bundle(ck['agent'])
    returns=[]
    for ep in range(args.episodes):
        obs,_=env.reset(seed=seed+1000+ep); planner.reset(); ret=0.0
        while True:
            plan=planner.plan(obs); base=plan_chunk(plan.actions,k)
            if agent is None: corrected=base
            else:
                c=np.concatenate([obs,base.reshape(-1)]); residual=agent.act(c,deterministic=True).reshape(k,ad)
                gate=adaptive_uncertainty_gate(plan.uncertainty,rc['gate_threshold'],rc['gate_temperature']) if method=='planning_residual' and rc.get('adaptive_gate',True) else rc.get('fixed_gate',1.0)
                corrected=base+gate*rc['residual_scale']*residual*span
            obs,r,te,tr,_=env.step(np.clip(corrected[0],low,high)); ret+=r
            if te or tr: break
        returns.append(ret); print(f'episode={ep} return={ret:.3f}')
    print(f'mean={np.mean(returns):.3f} std={np.std(returns):.3f}'); env.close()
if __name__=='__main__': main()
