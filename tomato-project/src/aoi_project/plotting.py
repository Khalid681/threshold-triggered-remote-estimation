"""Publication figures: every threshold has its own panel; node-cluster intervals."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .analysis import CONDITION

LABELS = dict(hold='Hold', full_drift='Full drift', weighted_drift='Weighted drift', local_linear='Local linear')


def figure_inputs(run_dir):
    root = Path(run_dir)
    tables = {name:pd.read_csv(root/'summaries'/f'{name}_summary.csv') for name in ['accuracy','channel','freshness']}
    raw_channel = pd.read_csv(root/'raw'/'channel_results.csv')
    return tables, sorted(raw_channel.node_id.unique().tolist())


def _save(fig, output, name):
    fig.tight_layout()
    for extension in ['png','pdf']:
        fig.savefig(output/f'{name}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _curve(ax, frame, x, y, label, x_ci=False):
    g = frame.sort_values('loss_probability')
    xx, yy = g[x].to_numpy(float), g[y+'_mean'].to_numpy(float)
    line, = ax.plot(xx, yy, 'o-', label=label, markersize=4)
    low, high = g[y+'_ci_low'].to_numpy(float), g[y+'_ci_high'].to_numpy(float)
    # Draw the actual percentile limits even when a skewed bootstrap excludes its point estimate.
    ax.vlines(xx, low, high, color=line.get_color(), alpha=.75, linewidth=1.2)
    if x_ci:
        stem = x.removesuffix('_mean')
        ax.hlines(yy, g[stem+'_ci_low'], g[stem+'_ci_high'], color=line.get_color(), alpha=.75, linewidth=1.2)


def plot_run(run_dir):
    root = Path(run_dir)
    output = root/'figures'
    output.mkdir(exist_ok=True)
    tables, nodes = figure_inputs(root)
    a,c,f = (tables[name] for name in ['accuracy','channel','freshness'])
    common = CONDITION+['feedback_model','channel_model']
    rate = 'delivered_update_rate_per_minute'
    rates = c[common+[rate+'_mean',rate+'_ci_low',rate+'_ci_high']]
    trade_accuracy = a.merge(rates, on=common, validate='many_to_one')
    trade_fresh = f.merge(rates, on=common, validate='many_to_one')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    jobs = [
        ('mse_loss',a,'loss_probability','mse','MSE','estimator'),
        ('nmse_loss',a,'loss_probability','nmse','NMSE (test variance)','estimator'),
        ('aoi_loss',c,'loss_probability','mean_aoi_minutes','Mean AoI (minutes)',None),
        ('freshness_loss',f,'loss_probability','freshness_violation_rate','Time above deadline','freshness_deadline_multiplier'),
        ('mse_rate',trade_accuracy,rate+'_mean','mse','MSE','estimator'),
        ('aoi_rate',c,rate+'_mean','mean_aoi_minutes','Mean AoI (minutes)',None),
        ('freshness_rate',trade_fresh,rate+'_mean','freshness_violation_rate','Time above deadline','freshness_deadline_multiplier')]
    for signal in sorted(a.signal.unique()):
        thresholds = sorted(a.threshold_multiplier.unique())
        for name, table, x, y, ylabel, group in jobs:
            fig, axes = plt.subplots(1,len(thresholds),figsize=(4.2*len(thresholds),4),squeeze=False)
            for ax, threshold in zip(axes[0],thresholds):
                frame = table[(table.signal==signal)&(table.threshold_multiplier==threshold)]
                curves = frame.groupby(group, sort=True) if group else [('Channel policy',frame)]
                for label, g in curves:
                    label = LABELS.get(label,str(label)+'x deadline' if group=='freshness_deadline_multiplier' else str(label))
                    _curve(ax,g,x,y,label,x_ci=x!= 'loss_probability')
                ax.set(title=f'Threshold multiplier {threshold:g}', ylabel=ylabel, xlabel='Loss probability' if x=='loss_probability' else 'Delivered updates / minute')
                ax.grid(alpha=.2)
            axes[0][-1].legend(fontsize=8)
            fig.suptitle(f'{signal} | {name.replace("_"," ")} | 95% node bootstrap intervals')
            _save(fig,output,f'{signal.lower()}_{name}')
        fig, axes = plt.subplots(1,len(thresholds),figsize=(4.2*len(thresholds),4),squeeze=False)
        for ax, threshold in zip(axes[0], thresholds):
            frame = c[(c.signal==signal)&(c.threshold_multiplier==threshold)]
            for kind in ['attempted','delivered']:
                _curve(ax,frame,'loss_probability',kind+'_update_rate_per_minute',kind.capitalize())
            ax.set(title=f'Threshold multiplier {threshold:g}',xlabel='Loss probability',ylabel='Updates / minute')
            ax.grid(alpha=.2)
        axes[0][-1].legend()
        fig.suptitle(f'{signal} | communication | 95% node bootstrap intervals')
        _save(fig,output,f'{signal.lower()}_update_rates')
    paired = pd.read_csv(root/'summaries'/'paired_comparison.csv')
    effects = pd.read_csv(root/'summaries'/'node_level_effects.csv')
    calibration = pd.read_csv(root/'raw'/'calibration_results.csv')
    for signal in sorted(a.signal.unique()):
        for name in ['win_tie_loss','beta_distribution','node_effects']:
            if name == 'node_effects':
                _plot_node_effects(effects, signal, nodes, output)
                continue
            thresholds = sorted(a.threshold_multiplier.unique())
            fig,axes = plt.subplots(1,len(thresholds),figsize=(4.6*len(thresholds),5),squeeze=False)
            for ax, threshold in zip(axes[0],thresholds):
                if name=='win_tie_loss':
                    g=paired[(paired.signal==signal)&(paired.threshold_multiplier==threshold)]
                    labels=[f'{r.loss_probability:g} / {LABELS[r.candidate]}' for r in g.itertuples()]
                    bottom=np.zeros(len(g))
                    for category,color in [('win','#228833'),('tie','#bbbbbb'),('loss','#cc6677')]:
                        ax.barh(np.arange(len(g)),g[category+'_fraction'],left=bottom,label=category,color=color)
                        bottom+=g[category+'_fraction'].to_numpy()
                    ax.set_yticks(np.arange(len(g)),labels,fontsize=7)
                    ax.set_xlabel('Descriptive fraction of unique pairs')
                elif name=='beta_distribution':
                    g=calibration[(calibration.signal==signal)&(calibration.threshold_multiplier==threshold)]
                    losses=sorted(g.loss_probability.unique())
                    for j,(column,color) in enumerate([('beta','#4477aa'),('local_beta','#ee6677')]):
                        arrays=[g[g.loss_probability==p][column].to_numpy() for p in losses]
                        box=ax.boxplot(arrays,positions=np.arange(len(losses))+j*.3, widths=.25,patch_artist=True,manage_ticks=False)
                        for patch in box['boxes']:
                            patch.set_facecolor(color)
                        ax.plot([],[],color=color,label='Weighted drift' if column=='beta' else 'Local linear')
                    ax.set_xticks(np.arange(len(losses))+.15,[f'{p:g}' for p in losses])
                    ax.set(xlabel='Loss probability',ylabel='Validation-selected beta (segment distribution)',ylim=(-.05,1.05))
                ax.set_title(f'Threshold multiplier {threshold:g}')
                ax.grid(alpha=.15)
            axes[0][-1].legend(fontsize=6)
            fig.suptitle(f'{signal} | {name.replace("_"," ")} (descriptive)')
            _save(fig,output,f'{signal.lower()}_{name}')
    manifest = dict(node_ids=nodes, thresholds=sorted(a.threshold_multiplier.unique().tolist()),
                    deadline_multipliers=sorted(f.freshness_deadline_multiplier.unique().tolist()),
                    uncertainty='10,000 resamples of equal-weight node means; descriptive distribution/effect figures are labeled',
                    files=sorted(p.name for p in output.iterdir() if p.suffix in ['.png','.pdf']))
    (root/'reports'/'figure_inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def _plot_node_effects(effects, signal, nodes, output):
    subset = effects[effects.signal == signal]
    thresholds = sorted(subset.threshold_multiplier.unique())
    losses = sorted(subset.loss_probability.unique())
    fig, axes = plt.subplots(len(losses), len(thresholds), figsize=(4.2*len(thresholds), 3.2*len(losses)), squeeze=False)
    for i, loss in enumerate(losses):
        for j, threshold in enumerate(thresholds):
            ax = axes[i,j]
            frame = subset[(subset.loss_probability==loss)&(subset.threshold_multiplier==threshold)]
            for candidate, group in frame.groupby('candidate',sort=True):
                ax.plot(group.node_id, group.mse_difference, 'o-', markersize=3, label=LABELS[candidate])
            ax.axhline(0,color='black',linewidth=.8)
            ax.set(title=f'Threshold {threshold:g} | loss {loss:g}', xlabel='Node ID', ylabel='Candidate MSE minus hold MSE')
            ax.set_xticks(nodes)
            ax.grid(alpha=.2)
    axes[0,-1].legend(fontsize=8)
    fig.suptitle(f'{signal} | effects by node (descriptive; negative favors prediction)')
    _save(fig,output,f'{signal.lower()}_node_effects')
