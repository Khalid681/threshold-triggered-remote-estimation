from __future__ import annotations
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aoi_project.baseline import run_baseline_sweep, generate_random_walk, simulate_loss_free_hold
from aoi_project.provenance import prepare_run, write_metadata, finish_run


def main():
    parser=argparse.ArgumentParser(description='Symmetric random-walk benchmark (discrete AoI convention)')
    parser.add_argument('--slots',type=int,default=1000000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--output',default='results/runs')
    parser.add_argument('--run-name')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    output=prepare_run(args.output,args.run_name,args.overwrite)
    metadata=write_metadata(output,dict(baseline_slots=args.slots,seed=args.seed))
    results=run_baseline_sweep(list(range(1,11)),args.slots,args.seed)
    # Independent contiguous increment blocks give descriptive MC uncertainty, not node uncertainty.
    states=generate_random_walk(args.slots,.5,.5,args.seed)
    blocks=np.array_split(states,20)
    rng=np.random.default_rng(2026)
    indices=rng.integers(0,20,(10000,20))
    for threshold in results.threshold:
        values=[simulate_loss_free_hold(block,int(threshold)) for block in blocks]
        for metric in ['update_rate','mean_aoi','mse']:
            samples=np.asarray([v[metric] for v in values])[indices].mean(axis=1)
            lo,hi=np.quantile(samples,[.025,.975])
            results.loc[results.threshold==threshold,metric+'_mc_ci_low']=lo
            results.loc[results.threshold==threshold,metric+'_mc_ci_high']=hi
    results.to_csv(output/'raw'/'baseline_validation.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric,label in zip(axes,['update_rate','mean_aoi','mse'],['Update rate','Discrete AoI (slots)','MSE']):
        ax.plot(results.threshold,results['theoretical_'+metric],'k--',label='Analytical')
        ax.plot(results.threshold,results[metric],'o-',label='Simulation')
        ax.vlines(results.threshold,results[metric+'_mc_ci_low'],results[metric+'_mc_ci_high'])
        ax.set(xlabel='Threshold',ylabel=label)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Symmetric random walk: 95% block Monte Carlo intervals')
    fig.tight_layout()
    fig.savefig(output/'figures'/'baseline_validation.png',dpi=300)
    fig.savefig(output/'figures'/'baseline_validation.pdf')
    plt.close(fig)
    checks=dict(finite=bool(np.isfinite(results.to_numpy()).all()),rows=len(results)==10,slots=args.slots,
                max_relative_error={m:float(((results[m]-results['theoretical_'+m]).abs()/results['theoretical_'+m].replace(0,np.nan)).max()) for m in ['update_rate','mean_aoi','mse']})
    checks['analytical_agreement_within_5_percent'] = all(v < .05 for v in checks['max_relative_error'].values())
    checks['passed'] = checks['finite'] and checks['rows'] and checks['analytical_agreement_within_5_percent']
    (output/'logs'/'baseline.txt').write_text(results.to_string(index=False)+'\n')
    (output/'reports'/'baseline_quality.json').write_text(json.dumps(checks,indent=2)+'\n')
    finish_run(output,metadata)
    print(results.to_string(index=False))
    print(f'Run: {output}')


if __name__=='__main__':
    main()
