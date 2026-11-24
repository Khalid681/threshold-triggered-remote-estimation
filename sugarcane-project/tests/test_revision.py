import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from aoi_project.analysis import (KEY, cluster_interval, paired_comparison, fdr_bh, sign_flip_pvalue, summarize, node_means)
from aoi_project.channel import (ReceptionTrace, derive_seed, simulate_threshold_channel, loss_stream, burst_transition_probabilities)
from aoi_project.data import TraceSplit, chronological_split
from aoi_project.estimators import estimate_from_receptions
from aoi_project.experiments import run_agriculture_sweep, save_results, validate_config
from aoi_project.metrics import freshness_metrics, communication_metrics, aggregate_pdr, estimation_metrics
from aoi_project.provenance import prepare_run


def config():
    return dict(data_path='data/Manzano2022COMPAG_data_cleaned.csv', signals=['Temp','Moist'], nodes=list(range(1,13)),
                threshold_multipliers=[.5,3.], loss_probabilities=[0.,.3], calibration_seeds=[1001,1002], evaluation_seeds=[101,102,103],
                beta_grid=[0.,.5,1.], freshness_deadline_multipliers=[2.,3.,6.], feedback_model='ack', channel_model='bernoulli',
                mean_bad_burst_length=5., delivery_delay_minutes=0., split_gap_multiplier=3., minimum_segment_records=100,
                train_fraction=.6, validation_fraction=.2)


def split(node=1, segment=0, constant=False):
    values=np.zeros(50) if constant else np.sin(np.arange(50)/3)+np.arange(50)*.05
    frame=pd.DataFrame(dict(timestamp=pd.date_range('2022-01-01',periods=50,freq='20min'),value=values))
    a,b,c=chronological_split(frame)
    return TraceSplit(node,'Temp',segment,20.,a,b,c)


def simple_trace(times, generation):
    n=len(times)
    return ReceptionTrace(np.array(times,float),np.zeros(n),np.zeros(n),np.array(generation,float),np.zeros(n,bool),np.zeros(n,bool))


def test_continuous_aoi_exact():
    m=freshness_metrics(simple_trace([0,10,20,30],[0,0,20,20]),5)
    assert m['mean_aoi_minutes']==pytest.approx(25/3)
    assert m['freshness_violation_rate']==pytest.approx(2/3)
    assert m['peak_aoi_minutes']==20
    assert m['sample_mean_aoi_minutes']==5


@pytest.mark.parametrize('deadline,expected',[(5,.5),(10,0.),(2,.8)])
def test_deadline_crossing_within_interval(deadline,expected):
    assert freshness_metrics(simple_trace([0,10],[0,10]),deadline)['freshness_violation_rate']==pytest.approx(expected)


def test_peak_before_reset():
    assert freshness_metrics(simple_trace([0,10],[0,10]),1)['peak_aoi_minutes']==10


def test_irregular_continuous_aoi():
    m=freshness_metrics(simple_trace([0,2,7],[0,0,7]),3)
    assert m['mean_aoi_minutes']==3.5
    assert m['freshness_violation_rate']==pytest.approx(4/7)


def test_zero_attempt_pdr():
    assert np.isnan(communication_metrics(simple_trace([0,10],[0,0]))['packet_delivery_ratio'])


def test_aggregate_pdr_is_ratio_of_totals():
    frame=pd.DataFrame(dict(attempted_packets=[0,1,9],delivered_packets=[0,1,0]))
    assert aggregate_pdr(frame)==.1
    assert np.isnan(aggregate_pdr(frame.iloc[:1]))


def test_delivery_subset_validation():
    t=simple_trace([0,1],[0,0])
    t.delivered[1]=True
    with pytest.raises(ValueError):
        communication_metrics(t)


def test_seed_stable_and_identity_sensitive():
    identity=[101,'evaluation',1,'Temp',0,.5,.3,'ack','bernoulli']
    original=derive_seed(*identity)
    assert original==derive_seed(*identity)
    for index,value in [(0,102),(1,'calibration'),(2,2),(3,'Moist'),(4,1),(5,1.),(6,.2),(7,'no_ack'),(8,'gilbert_elliott')]:
        changed=identity.copy(); changed[index]=value
        assert derive_seed(*changed)!=original
    command='from aoi_project.channel import derive_seed; print(derive_seed(101,"evaluation",1,"Temp",0,.5,.3,"ack","bernoulli"))'
    assert int(subprocess.check_output([sys.executable,'-c',command],text=True))==original


def test_nodes_have_independent_loss_streams():
    a=derive_seed(1,'evaluation',1,'Temp',0,1,.3,'ack','bernoulli')
    b=derive_seed(1,'evaluation',2,'Temp',0,1,.3,'ack','bernoulli')
    assert not np.array_equal(loss_stream(1000,.3,a),loss_stream(1000,.3,b))


def test_seed_sets_reject_overlap():
    c=config(); c['calibration_seeds']=[101]
    with pytest.raises(ValueError,match='disjoint'):
        validate_config(c)


def test_delay_rejected():
    with pytest.raises(ValueError,match='arrival'):
        simulate_threshold_channel([0,1],[0,2],1,0,1,delivery_delay_minutes=.5)


def test_normalized_constant_trace_guard():
    m=estimation_metrics(np.ones(4),np.zeros(4))
    assert m['constant_trace'] and np.isnan(m['nmse']) and m['mse']==1


def test_local_linear_receiver_only_and_clipped():
    t=simulate_threshold_channel([0,1,2,3],[0,2,2,2],1,0,1)
    estimate=estimate_from_receptions(t,'local_linear',beta=.5,slope_clip=1)
    np.testing.assert_allclose(estimate,[0,2,2.5,3])
    t.values[2:]=1000  # Unreceived ground truth cannot change predictions.
    np.testing.assert_allclose(estimate_from_receptions(t,'local_linear',beta=.5,slope_clip=1),estimate)


def test_clipping_and_calibration_training_only():
    from aoi_project.experiments import calibrate_split
    s=split(); before=calibrate_split(s,.5,.3,config())
    s.test['value']=100000.
    after=calibrate_split(s,.5,.3,config())
    assert before==after


def test_normalized_tables_and_shared_trace():
    c=config(); tables=run_agriculture_sweep([split(1),split(2)],c)
    assert len(tables['channel'])==2*2*(1+3)
    assert len(tables['estimation'])==4*len(tables['channel'])
    assert len(tables['freshness'])==3*len(tables['channel'])
    assert 'freshness_deadline_minutes' not in tables['estimation']
    assert tables['estimation'].groupby(KEY).trace_sha256.nunique().eq(1).all()
    assert not tables['estimation'].duplicated(KEY+['estimator']).any()
    zero=tables['channel'].query('loss_probability == 0')
    assert zero.seed.nunique()==1 and len(zero)==4
    paired,nodes=paired_comparison(tables['estimation'])
    assert set(paired[paired.loss_probability==0].paired_runs)=={2}
    assert set(paired[paired.loss_probability>.0].paired_runs)=={6}
    assert len(nodes)==2*2*2*3


def paired_fixture():
    rows=[]
    for node in range(1,4):
        for segment in range(node):
            for seed in [1,2]:
                for name,mse in [('hold',2.),('weighted_drift',float(node)),('full_drift',4.)]:
                    rows.append(dict(node_id=node,signal='Temp',segment_id=segment,threshold_multiplier=1.,loss_probability=.3,seed=seed,estimator=name,mse=mse))
    return pd.DataFrame(rows)


def test_legacy_deadlines_never_increase_pairs():
    df=paired_fixture()
    legacy=pd.concat([df.assign(freshness_deadline_minutes=d) for d in [10,20,30]])
    paired,_=paired_comparison(legacy)
    assert paired.paired_runs.tolist()==[12,12]
    assert paired.nodes.tolist()==[3,3]
    assert paired.segments.tolist()==[6,6]
    assert paired.loc[paired.candidate=='weighted_drift','mean_mse_difference'].iloc[0]==0


def test_conflicting_legacy_duplicate_rejected():
    df=paired_fixture()
    conflict=df.iloc[:1].copy(); conflict['mse']=999
    with pytest.raises(ValueError,match='Conflicting'):
        paired_comparison(pd.concat([df,conflict]))


def test_ratio_of_means_not_mean_percentages():
    df=paired_fixture()
    df.loc[(df.node_id==1)&(df.estimator=='hold'),'mse']=0
    result,_=paired_comparison(df)
    row=result[result.candidate=='weighted_drift'].iloc[0]
    assert row.aggregate_relative_improvement_percent==pytest.approx(-50.)


def test_bootstrap_reproducible_with_independent_nodes():
    a=cluster_interval([1,2,9]); b=cluster_interval([1,2,9])
    assert a==b
    assert np.isnan(cluster_interval([1])[0])
    assert cluster_interval([0,0,0])==(0,0)


def test_nested_weighting_equal_nodes():
    g=pd.DataFrame(dict(node_id=[1,1,1,2],segment_id=[0,0,1,0],mse=[0.,0.,10.,20.]))
    assert node_means(g,['mse']).mse.tolist()==[5.,20.]


def test_fdr_known_values():
    np.testing.assert_allclose(fdr_bh([.01,.04,.03,.002]),[.02,.04,.04,.008])
    assert np.isnan(fdr_bh([np.nan])[0])


def test_exact_sign_flip():
    assert sign_flip_pvalue([1]*12)==2/4096
    assert sign_flip_pvalue([0]*12)==1


def test_summary_group_counts_and_deadline_multipliers(tmp_path):
    tables=run_agriculture_sweep([split(1),split(2)],config())
    save_results(tables,tmp_path)
    assert len(pd.read_csv(tmp_path/'summaries'/'accuracy_summary.csv'))==2*2*4
    fresh=pd.read_csv(tmp_path/'summaries'/'freshness_summary.csv')
    assert len(fresh)==2*2*3
    assert set(fresh.freshness_deadline_multiplier)=={2,3,6}


@pytest.mark.parametrize('probability',[.1,.3,.5])
def test_gilbert_elliott_stationary_loss_and_burst_length(probability):
    losses=loss_stream(200000,probability,2026,'gilbert_elliott',5)
    assert abs(losses.mean()-probability)<.012
    edges=np.diff(np.r_[False,losses,False].astype(int))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    assert abs(lengths.mean()-5)<.25


def test_burst_parameters_and_zero_loss():
    a,b=burst_transition_probabilities(.3,5)
    assert a/(a+b)==pytest.approx(.3)
    assert not loss_stream(1000,0,1,'gilbert_elliott',5).any()
    with pytest.raises(ValueError):
        burst_transition_probabilities(.9,1)


def test_chronological_no_overlap():
    s=split()
    assert s.train.timestamp.max()<s.validation.timestamp.min()<s.test.timestamp.min()
    assert s.validation.timestamp.max()<s.test.timestamp.min()


def test_run_collision_preserves_data(tmp_path):
    output=prepare_run(tmp_path,'fixed'); (output/'marker').write_text('old')
    with pytest.raises(FileExistsError):
        prepare_run(tmp_path,'fixed')
    prepare_run(tmp_path,'fixed',True)
    assert len(list(tmp_path.glob('fixed.previous-*/marker')))==1
    with pytest.raises(ValueError):
        prepare_run(tmp_path,'../unsafe')


def test_all_node_figure_input(tmp_path):
    from aoi_project.plotting import figure_inputs
    tables=run_agriculture_sweep([split(n) for n in range(1,13)],config())
    save_results(tables,tmp_path)
    frames,nodes=figure_inputs(tmp_path)
    assert nodes==list(range(1,13))
    assert frames['freshness'].nodes.eq(12).all()


def test_complete_quick_integration(tmp_path):
    from aoi_project.quality import check_run
    command=[sys.executable,'run_agriculture.py','--quick','--run-name','integration','--output',str(tmp_path)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    root=tmp_path/'integration'
    report=check_run(root)
    assert report['passed'],report['checks']
    assert report['node_ids']==list(range(1,13))
    subprocess.run([sys.executable,'paired_analysis.py','--run-dir',str(root)],check=True,capture_output=True)
    subprocess.run([sys.executable,'plot_results.py','--run-dir',str(root)],check=True,capture_output=True)
    assert check_run(root)['passed']
    assert len(list((root/'figures').glob('*.pdf')))==22
    df=pd.read_csv(root/'raw'/'estimation_results.csv')
    df.loc[0,'mse']=np.inf
    df.to_csv(root/'raw'/'estimation_results.csv',index=False)
    assert not check_run(root)['passed']


def test_1200_pairs_not_3600_deadline_copies():
    rows=[]
    for segment in range(40):
        for seed in range(30):
            for deadline in [2.,3.,6.]:
                for name,mse in [('hold',1.),('weighted_drift',.9),('full_drift',1.2)]:
                    rows.append(dict(node_id=segment%12+1,signal='Temp',segment_id=segment,threshold_multiplier=1.,
                                     loss_probability=.1,seed=seed,freshness_deadline_multiplier=deadline,estimator=name,mse=mse))
    comparisons,_=paired_comparison(pd.DataFrame(rows))
    assert comparisons.paired_runs.eq(1200).all()
    assert comparisons.nodes.eq(12).all()
    assert comparisons.segments.eq(40).all()


@pytest.mark.parametrize('feedback,channel',[('no_ack','bernoulli'),('ack','gilbert_elliott')])
def test_optional_scenarios_preserve_paired_design(feedback,channel):
    cfg=config(); cfg.update(feedback_model=feedback,channel_model=channel)
    tables=run_agriculture_sweep([split()],cfg)
    assert tables['estimation'].groupby(KEY).trace_sha256.nunique().eq(1).all()
    assert (tables['channel'].delivered_packets<=tables['channel'].attempted_packets).all()
    if feedback=='no_ack':
        assert tables['channel'].retry_attempts.eq(0).all()
        assert tables['channel'].retry_fraction.isna().all()
