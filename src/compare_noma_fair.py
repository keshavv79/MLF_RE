"""
compare_noma_fair.py
====================================================================
Eval + ablation for the M=4 NOMA v3-FAIR policy (weighted BF + soft null).
NOMA counterpart of compare_oma_v3_easy.py.

Policy output ("alpha") is the NOMA power split (a_n, a_f, a_T).
Physics imported from train_noma_fair_maml (fair beamforming).

Methods (main)
  baseline    - random RIS phi + fixed fair alloc (a_n=0.20, a_f=0.55, a_T=0.25)
  noma_fair   - MAML bilevel DL (inner-loop phi + AlphaNet power split)

Ablation (decoupling phi vs power-split optimisation)
  randPhi_randAlpha  - random phi + random-softmax alpha
  randPhi_fairAlpha  - random phi + fixed fair alloc  (= baseline)
  randPhi_optAlpha   - random phi + AlphaNet alpha (no inner loop)
  randAlpha_optPhi   - random frozen alpha + inner-loop optimised phi
  fullOptimized      - noma_fair

Outputs (under --out-dir, default ./eval_noma_fair/)
  rate_hist.png, rate_cdf.png, per_leg_rate_cdf.png, alloc_hist.png,
  qos_violations.png, improvement_bars.png, theta_heatmap.png,
  rate_vs_qos_margin.png, summary.csv, summary.txt   (+ ablation/ subfolder)
"""

import argparse, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_dl_v3_easy import load_dataset
from train_noma_fair_maml import (
    AlphaNet, inner_optimize,
    physics_step, rates_from_gains, make_features, lagrangian_loss,
)

FAIR_ALLOC = (0.20, 0.55, 0.25)   # (a_n, a_f, a_T)


class _Args:
    def __init__(self, d):
        defaults = dict(w_c=0.7, w_s=0.3, inner_steps=10, inner_lr=0.1,
                        gamma_theta=0.3,
                        lam_c=2.0, lam_s=50.0, lam_n=15.0, lam_f=20.0,
                        R_th_n=1.0, R_th_f=1.0)
        for k, v in defaults.items(): setattr(self, k, v)
        for k, v in d.items(): setattr(self, k, v)


def collect_per_channel(forward_fn, loader, device, R_th_c, R_th_s):
    Rn, Rf, Rs, an, af, aT, th = [], [], [], [], [], [], []
    for batch in loader:
        batch = tuple(b.to(device) for b in batch)
        R_n, R_f, R_s, alpha, theta = forward_fn(*batch)
        Rn.append(R_n.detach().cpu().numpy()); Rf.append(R_f.detach().cpu().numpy())
        Rs.append(R_s.detach().cpu().numpy())
        an.append(alpha[:, 0].detach().cpu().numpy()); af.append(alpha[:, 1].detach().cpu().numpy())
        aT.append(alpha[:, 2].detach().cpu().numpy())
        th.append(torch.angle(theta).detach().cpu().numpy())
    Rn=np.concatenate(Rn); Rf=np.concatenate(Rf); Rs=np.concatenate(Rs)
    an=np.concatenate(an); af=np.concatenate(af); aT=np.concatenate(aT)
    theta_ang=np.concatenate(th, axis=0)
    qc=((Rn+Rf)<R_th_c).astype(np.float32); qs=(Rs<R_th_s).astype(np.float32)
    qany=((qc+qs)>0).astype(np.float32)
    return dict(Rn=Rn, Rf=Rf, Rs=Rs, an=an, af=af, aT=aT, theta_ang=theta_ang,
                qc=qc, qs=qs, qany=qany,
                Rsum_paper=0.7*(Rn+Rf)+0.3*Rs, Rsum_bal=0.5*(Rn+Rf)+0.5*Rs)


def _theta(phi, dt): return torch.complex(torch.cos(phi), torch.sin(phi)).to(dt)


def build_baseline_fn(cfg, device, seed=0):
    torch.manual_seed(seed)
    @torch.no_grad()
    def fw(H_BR, h_RDn, h_RDf, h_RT, h_TR):
        B = H_BR.size(0)
        phi = 2*np.pi*torch.rand(B, cfg['N'], device=device)
        gains = physics_step(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
        alpha = torch.tensor(FAIR_ALLOC, device=device).expand(B, 3)
        R_n, R_f, R_s = rates_from_gains(gains, alpha, cfg['P_tot'], cfg['sigma2'])
        return R_n, R_f, R_s, alpha, _theta(phi, H_BR.dtype)
    return fw


def build_randPhi_randAlpha_fn(cfg, device, seed=1):
    torch.manual_seed(seed)
    @torch.no_grad()
    def fw(H_BR, h_RDn, h_RDf, h_RT, h_TR):
        B = H_BR.size(0)
        phi = 2*np.pi*torch.rand(B, cfg['N'], device=device)
        gains = physics_step(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
        alpha = torch.softmax(torch.randn(B, 3, device=device), dim=-1)
        R_n, R_f, R_s = rates_from_gains(gains, alpha, cfg['P_tot'], cfg['sigma2'])
        return R_n, R_f, R_s, alpha, _theta(phi, H_BR.dtype)
    return fw


def build_randPhi_optAlpha_fn(ckpt, cfg, device, seed=2):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    net = AlphaNet(in_dim=9, hidden=c['args'].get('hidden', 256)).to(device)
    net.load_state_dict(c['state_dict']); net.eval()
    torch.manual_seed(seed)
    @torch.no_grad()
    def fw(H_BR, h_RDn, h_RDf, h_RT, h_TR):
        B = H_BR.size(0)
        phi = 2*np.pi*torch.rand(B, cfg['N'], device=device)
        gains = physics_step(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
        feats = make_features(gains, cfg['R_th_c'], cfg['R_th_s'], cfg['P_tot'], cfg['sigma2'])
        alpha = net(feats)
        R_n, R_f, R_s = rates_from_gains(gains, alpha, cfg['P_tot'], cfg['sigma2'])
        return R_n, R_f, R_s, alpha, _theta(phi, H_BR.dtype)
    return fw


def build_randAlpha_optPhi_fn(cfg, device, seed=3, inner_steps=10, inner_lr=0.1):
    torch.manual_seed(seed)
    def fw(H_BR, h_RDn, h_RDf, h_RT, h_TR):
        B = H_BR.size(0); dev = H_BR.device
        alpha = torch.softmax(torch.randn(B, 3, device=dev), dim=-1).detach()
        phi = 2*np.pi*torch.rand(B, cfg['N'], device=dev); phi.requires_grad_(True)
        with torch.enable_grad():
            inner_opt = torch.optim.Adam([phi], lr=inner_lr)
            for _ in range(inner_steps):
                gains = physics_step(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
                R_n, R_f, R_s = rates_from_gains(gains, alpha, cfg['P_tot'], cfg['sigma2'])
                L, _ = lagrangian_loss(R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
                                       lam_c=2.0, lam_s=50.0, lam_n=0.0, lam_f=0.0)
                phi.grad = torch.autograd.grad(L, phi, create_graph=False)[0]
                inner_opt.step()
            gains = physics_step(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
            R_n, R_f, R_s = rates_from_gains(gains, alpha, cfg['P_tot'], cfg['sigma2'])
        return R_n.detach(), R_f.detach(), R_s.detach(), alpha, _theta(phi.detach(), H_BR.dtype)
    return fw


def build_maml_fn(ckpt, cfg, device):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    net = AlphaNet(in_dim=9, hidden=c['args'].get('hidden', 256)).to(device)
    net.load_state_dict(c['state_dict']); net.eval()
    ia = _Args(c['args'])
    def fw(*batch):
        with torch.enable_grad():
            _, _, R_n, R_f, R_s, phi, alpha = inner_optimize(net, batch, cfg, ia, create_meta_graph=False)
        return R_n.detach(), R_f.detach(), R_s.detach(), alpha.detach(), _theta(phi.detach(), batch[0].dtype)
    return fw


COLORS = {'baseline':'#888888','noma_fair':'#d62728','randPhi_randAlpha':'#cccccc',
          'randPhi_fairAlpha':'#bcbd22','randPhi_optAlpha':'#ff7f0e',
          'randAlpha_optPhi':'#17becf','fullOptimized':'#d62728'}


def plot_rate_hist(res, out, key='Rsum_paper', sfx='(paper 0.7/0.3)'):
    plt.figure(figsize=(9,5.5))
    for n,r in res.items(): plt.hist(r[key],bins=50,alpha=0.55,label=n,color=COLORS.get(n),edgecolor='none')
    plt.xlabel('R_DL_sum (bits/s/Hz)'); plt.ylabel('Count')
    plt.title(f'NOMA-fair weighted sum-rate {sfx}'); plt.legend(); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(out,dpi=200); plt.close()


def plot_rate_cdf(res, out, key='Rsum_paper', sfx='(paper 0.7/0.3)'):
    plt.figure(figsize=(9,5.5))
    for n,r in res.items():
        xs=np.sort(r[key]); ys=np.linspace(0,1,xs.size); plt.plot(xs,ys,label=n,color=COLORS.get(n),lw=2)
    plt.xlabel('R_DL_sum (bits/s/Hz)'); plt.ylabel('CDF')
    plt.title(f'NOMA-fair sum-rate CDF {sfx}'); plt.legend(); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(out,dpi=200); plt.close()


def plot_per_leg_rate_cdf(res, out, R_th_c, R_th_s):
    fig,ax=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for a,key,lab,th in [(ax[0],'Rn','R_n (near)',None),(ax[1],'Rf','R_f (far)',None),(ax[2],'Rs','R_s (sensing)',R_th_s)]:
        for n,r in res.items():
            xs=np.sort(r[key]); ys=np.linspace(0,1,xs.size); a.plot(xs,ys,label=n,color=COLORS.get(n),lw=2)
        if th is not None: a.axvline(th,color='black',ls=':',alpha=0.6,label=f'R_th={th}')
        a.set_xlabel('Rate (bits/s/Hz)'); a.set_title(lab); a.grid(True,alpha=0.3); a.legend(fontsize=8)
    ax[0].set_ylabel('CDF')
    fig.suptitle(f'NOMA-fair per-leg rate CDFs (R_th_c={R_th_c}, R_th_s={R_th_s})',fontsize=11)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(out,dpi=200); plt.close()


def plot_alloc_hist(res, out):
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    for a,key,lab in [(ax[0],'an','a_n (near power)'),(ax[1],'af','a_f (far power)'),(ax[2],'aT','a_T (sensing power)')]:
        for n,r in res.items(): a.hist(r[key],bins=40,alpha=0.55,label=n,color=COLORS.get(n),edgecolor='none')
        a.set_xlabel('Fraction of total power'); a.set_title(lab); a.grid(True,alpha=0.3); a.legend(fontsize=8)
    ax[0].set_ylabel('Count'); fig.suptitle('Learned NOMA power allocation',fontsize=11)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(out,dpi=200); plt.close()


def plot_qos_violations(res, out):
    labs=list(res); qany=[res[k]['qany'].mean() for k in labs]
    qc=[res[k]['qc'].mean() for k in labs]; qs=[res[k]['qs'].mean() for k in labs]
    x=np.arange(len(labs)); w=0.27
    plt.figure(figsize=(10,5.5))
    plt.bar(x-w,qany,w,label='qos_viol_any',color='#444444')
    plt.bar(x,qc,w,label='qc (R_n+R_f<R_th_c)',color='#1f77b4')
    plt.bar(x+w,qs,w,label='qs (R_s<R_th_s)',color='#d62728')
    plt.xticks(x,labs,rotation=15); plt.ylabel('Violation fraction')
    plt.title('NOMA-fair QoS violation (lower better)'); plt.legend(); plt.grid(True,alpha=0.3,axis='y')
    for i,v in enumerate(qany): plt.text(i-w,v+0.005,f'{100*v:.1f}%',ha='center',fontsize=8)
    plt.tight_layout(); plt.savefig(out,dpi=200); plt.close()


def plot_improvement_bars(res, base, out):
    labs=[k for k in res if k!=base]
    mets=[('Rsum_paper','R_DL_sum',False),('Rn','R_n',False),('Rf','R_f',False),('Rs','R_s',False),
          ('qany','qos_viol_any',True),('qs','qs',True),('qc','qc',True)]
    bm={m:res[base][m].mean() for m,_,_ in mets}
    nm=len(labs); w=0.8/max(nm,1); x=np.arange(len(mets))
    plt.figure(figsize=(12,6))
    for i,n in enumerate(labs):
        d=[]
        for m,_,low in mets:
            b=bm[m]; v=res[n][m].mean()
            if abs(b)<1e-9: d.append(0.0); continue
            p=100*(v-b)/abs(b); p=-p if low else p; d.append(max(min(p,500),-500))
        plt.bar(x+(i-(nm-1)/2)*w,d,w,label=n,color=COLORS.get(n))
    plt.axhline(0,color='black',lw=0.7); plt.xticks(x,[l for _,l,_ in mets],rotation=15)
    plt.ylabel('% improvement vs baseline (cap +-500)')
    plt.title('NOMA-fair per-metric improvement'); plt.legend(); plt.grid(True,alpha=0.3,axis='y')
    plt.tight_layout(); plt.savefig(out,dpi=200); plt.close()


def plot_rate_vs_qos_margin(res, out, R_th_c, R_th_s, key='Rsum_paper', sfx='(paper 0.7/0.3)'):
    plt.figure(figsize=(10,6))
    for n,r in res.items():
        m=np.minimum((r['Rn']+r['Rf'])-R_th_c, r['Rs']-R_th_s)
        plt.scatter(m,r[key],s=6,alpha=0.35,label=n,color=COLORS.get(n))
    plt.axvline(0,color='black',lw=0.7,ls='--',alpha=0.7)
    plt.xlabel('Min QoS margin = min(R_n+R_f-R_th_c, R_s-R_th_s) [bits/s/Hz]')
    plt.ylabel('R_DL_sum (bits/s/Hz)'); plt.title(f'NOMA-fair rate vs QoS margin {sfx}')
    plt.legend(loc='lower right'); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(out,dpi=200); plt.close()


def plot_theta_heatmap(res, out):
    names=list(res); n=len(names)
    fig,ax=plt.subplots(n,1,figsize=(9,1.0+0.8*n),squeeze=False)
    for i,nm in enumerate(names):
        ang=res[nm]['theta_ang']; ma=np.angle(np.exp(1j*ang).mean(axis=0))
        a=ax[i,0]; a.imshow(ma[np.newaxis,:],aspect='auto',cmap='twilight',vmin=-np.pi,vmax=np.pi)
        a.set_yticks([]); a.set_ylabel(nm,rotation=0,ha='right',va='center',fontsize=9)
        if i==n-1: a.set_xlabel('RIS element index')
        else: a.set_xticks([])
    fig.suptitle('Mean RIS phase per element (circular mean)',fontsize=10)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(out,dpi=200); plt.close()


def write_summary(res, base, R_th_c, R_th_s, out_dir):
    hdr=['method','R_DL_sum_paper','R_DL_sum_balanced','R_n','R_f','R_s',
         'qos_viol_any','qc','qs','a_n','a_f','a_T']
    with (out_dir/'summary.csv').open('w') as f:
        f.write(','.join(hdr)+'\n')
        for n,r in res.items():
            f.write(','.join([n,f"{r['Rsum_paper'].mean():.4f}",f"{r['Rsum_bal'].mean():.4f}",
                f"{r['Rn'].mean():.4f}",f"{r['Rf'].mean():.4f}",f"{r['Rs'].mean():.4f}",
                f"{r['qany'].mean():.4f}",f"{r['qc'].mean():.4f}",f"{r['qs'].mean():.4f}",
                f"{r['an'].mean():.4f}",f"{r['af'].mean():.4f}",f"{r['aT'].mean():.4f}"])+'\n')
    with (out_dir/'summary.txt').open('w') as f:
        f.write(f"NOMA v3-FAIR test-set comparison\nSamples: {next(iter(res.values()))['Rn'].size}\n")
        f.write(f"R_th_c={R_th_c}  R_th_s={R_th_s}\n\n")
        f.write(f"{'Method':<18}{'Rsum(7/3)':>12}{'R_n':>8}{'R_f':>8}{'R_s':>8}{'qany':>9}\n")
        f.write('-'*63+'\n')
        for n,r in res.items():
            f.write(f"{n:<18}{r['Rsum_paper'].mean():>12.4f}{r['Rn'].mean():>8.3f}"
                    f"{r['Rf'].mean():>8.3f}{r['Rs'].mean():>8.3f}{r['qany'].mean():>9.4f}\n")
    print(f"Saved summary -> {out_dir/'summary.csv'}")


def render_all(res, out_dir, R_th_c, R_th_s, base, sfx):
    plot_rate_hist(res,out_dir/'rate_hist.png','Rsum_paper',sfx)
    plot_rate_cdf(res,out_dir/'rate_cdf.png','Rsum_paper',sfx)
    plot_per_leg_rate_cdf(res,out_dir/'per_leg_rate_cdf.png',R_th_c,R_th_s)
    plot_alloc_hist(res,out_dir/'alloc_hist.png')
    plot_qos_violations(res,out_dir/'qos_violations.png')
    plot_improvement_bars(res,base,out_dir/'improvement_bars.png')
    plot_theta_heatmap(res,out_dir/'theta_heatmap.png')
    plot_rate_vs_qos_margin(res,out_dir/'rate_vs_qos_margin.png',R_th_c,R_th_s,'Rsum_paper',sfx)
    write_summary(res,base,R_th_c,R_th_s,out_dir)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mat', default='ISAC_RIS_NOMA_channels_v3_fair_TEST.mat')
    p.add_argument('--out-dir', default='eval_noma_fair')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    p.add_argument('--ckpt_maml', default='policy_noma_fair_best.pt')
    args=p.parse_args()

    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    ds=load_dataset(Path(args.mat))
    cfg=dict(N=ds['N'],M=ds['M'],P_tot=ds['P_tot'],sigma2=ds['sigma2'],beta_T=ds['beta_T'],
             R_th_c=ds['R_th_c'],R_th_s=ds['R_th_s'])
    R_th_c,R_th_s=ds['R_th_c'],ds['R_th_s']
    tensors=tuple(torch.from_numpy(ds[k]) for k in ['H_BR','h_RDn','h_RDf','h_RT','h_TR'])
    loader=DataLoader(TensorDataset(*tensors),batch_size=args.batch)
    print(f"NOMA-fair test: {args.mat} ({tensors[0].shape[0]} ch, N={ds['N']}, M={ds['M']})  R_th_c={R_th_c} R_th_s={R_th_s}\n")

    res={}
    print('baseline (random phi + fair alloc 0.20/0.55/0.25) ...')
    res['baseline']=collect_per_channel(build_baseline_fn(cfg,device),loader,device,R_th_c,R_th_s)
    if os.path.exists(args.ckpt_maml):
        print(f'noma_fair MAML from {args.ckpt_maml} ...')
        res['noma_fair']=collect_per_channel(build_maml_fn(args.ckpt_maml,cfg,device),loader,device,R_th_c,R_th_s)
    else:
        print(f'[warn] {args.ckpt_maml} not found; skipping MAML')

    print('main plots ...'); render_all(res,out_dir,R_th_c,R_th_s,'baseline','(paper 0.7/0.3)')

    if os.path.exists(args.ckpt_maml):
        print('ablation ...')
        ab={}
        ab['randPhi_randAlpha']=collect_per_channel(build_randPhi_randAlpha_fn(cfg,device),loader,device,R_th_c,R_th_s)
        ab['randPhi_fairAlpha']=res['baseline']
        ab['randPhi_optAlpha']=collect_per_channel(build_randPhi_optAlpha_fn(args.ckpt_maml,cfg,device),loader,device,R_th_c,R_th_s)
        ab['randAlpha_optPhi']=collect_per_channel(build_randAlpha_optPhi_fn(cfg,device),loader,device,R_th_c,R_th_s)
        ab['fullOptimized']=res['noma_fair']
        ab_dir=out_dir/'ablation'; ab_dir.mkdir(exist_ok=True)
        render_all(ab,ab_dir,R_th_c,R_th_s,'randPhi_randAlpha','(ablation, paper 0.7/0.3)')
        print('\nABLATION:')
        for n,r in ab.items():
            print(f"  {n:<20} Rsum={r['Rsum_paper'].mean():7.3f} R_n={r['Rn'].mean():6.2f} "
                  f"R_f={r['Rf'].mean():5.2f} R_s={r['Rs'].mean():5.2f} qany={r['qany'].mean():.3f}")

    print('\nHEADLINE:')
    for n,r in res.items():
        print(f"  {n:<18} Rsum={r['Rsum_paper'].mean():7.3f} qany={r['qany'].mean():.3f}")
    print(f"\nOutputs -> {out_dir.resolve()}")


if __name__ == '__main__':
    main()
