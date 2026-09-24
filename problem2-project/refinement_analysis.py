"""Report the frozen refinement, including negative findings and paired uncertainty."""
import argparse,csv,json
from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from refinement_run import weights,writecsv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='outputs/refinement_v2');args=ap.parse_args()
    root=Path(__file__).resolve().parent;out=root/args.run_dir
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    cfg=read('protocol.json');frozen=read('frozen_selection.json');raw=joblib.load(out/'validation_predictions.joblib')
    with (out/'validation_summary.csv').open(encoding='utf-8-sig') as f:summary=list(csv.DictReader(f))
    for row in summary:
        for k in row:
            if k!='model':row[k]=float(row[k])
    y=raw['classes'];r=raw['intensity'];names=frozen['scenario_names'];w=weights(cfg)
    video=np.array([sid.split('$_$')[0] for sid in raw['ids']]);uv=np.unique(video)
    group=[np.flatnonzero(video==v) for v in uv]
    preds={model:(np.array([raw['predictions'][model+'::'+name]['probabilities'] for name in names]).argmax(2),
                   np.array([raw['predictions'][model+'::'+name]['intensity'] for name in names])) for model in ['v1_seed17','v2_selected']}
    def score(model,counts):
        p,z=preds[model];n=counts.sum()
        idx=np.arange(len(names))[:,None]*9+y[None,:]*3+p
        cm=np.bincount(idx.ravel(),weights=np.broadcast_to(counts,idx.shape).ravel(),minlength=len(names)*9).reshape(-1,3,3)
        den=cm.sum(1)+cm.sum(2);diag=np.diagonal(cm,axis1=1,axis2=2)
        f1=np.divide(2*diag,den,out=np.zeros_like(den),where=den>0).mean(1)
        mae=np.abs(z-r[None,:])@counts/n
        return np.array([w@f1,w@mae,f1[0],mae[0]])
    rng=np.random.default_rng(3917);bootstrap=[]
    for _ in range(cfg['bootstrap_repeats']):
        ix=np.concatenate([group[i] for i in rng.integers(0,len(group),len(group))]);counts=np.bincount(ix,minlength=len(y))
        bootstrap.append(score('v2_selected',counts)-score('v1_seed17',counts))
    bs=np.array(bootstrap);point=score('v2_selected',np.ones(len(y)))-score('v1_seed17',np.ones(len(y)))
    ci=[]
    for i,name in enumerate(['weighted_macro_f1_delta','weighted_mae_delta','complete_macro_f1_delta','complete_mae_delta']):
        lo,hi=np.quantile(bs[:,i],[.025,.975]);ci.append(dict(quantity=name,delta=float(point[i]),ci_low=float(lo),ci_high=float(hi),video_clusters=len(group),repeats=cfg['bootstrap_repeats']))
    writecsv(out/'validation_paired_bootstrap.csv',ci)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    short={'v2_selected':'V2 selected','default_neutral_rule':'Neutral weight = 1','no_regression_calibration':'No intensity calibration',
           'ridge10_regression':'Ridge alpha = 10','shared_v1_fusion':'Shared V1 fusion','old_augmentation_mass':'Clean mass = 0.25',
           'v1_seed17':'V1 seed 17','v1_ensemble':'V1 ensemble'}
    for ax,key,title in zip(axes,['weighted_macro_f1','weighted_mae'],['Validation weighted Macro-F1 (higher better)','Validation weighted MAE (lower better)']):
        ax.barh([short[row['model']] for row in summary],[row[key] for row in summary],color=['#246b91']+['#91b6c8']*5+['#caa775']*2)
        ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'validation_component_comparison.png',dpi=170);plt.close(fig)
    new=next(row for row in summary if row['model']=='v2_selected');old=next(row for row in summary if row['model']=='v1_seed17')
    cs=frozen['classification'];rs=frozen['regression'];cv=read('cv_comparison.json')
    neutral_rows=[row for row in read('classification_search.json') if row['index']==cs['index']]
    neutral_table='\n'.join(f"| {row['neutral_weight']} | {row['weighted_macro_f1']:.4f} |" for row in neutral_rows)
    reg_rows=read('regression_search.json')
    regression_families=[min([row for row in reg_rows if row['spec']['kind']==kind],key=lambda row:row['weighted_mae']) for kind in ['ridge','absolute_svr']]
    regression_table='\n'.join(f"| {row['spec']['kind']} | {row['weighted_mae']:.4f} |" for row in regression_families)
    cvsummary=[]
    for model in ['v1_protocol_refit','v2_selected']:
        rr=[row for row in cv if row['model']==model]
        cvsummary.append({'model':model,'weighted_macro_f1':float(w@[row['f1_macro'] for row in rr]),'weighted_mae':float(w@[row['mae'] for row in rr])})
    js=lambda name,value:(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    js('cv_summary.json',cvsummary)
    table='\n'.join(f"| {short[row['model']]} | {row['complete_macro_f1']:.4f} | {row['complete_mae']:.4f} | {row['weighted_macro_f1']:.4f} | {row['weighted_mae']:.4f} |" for row in summary)
    cit='\n'.join(f"| {row['quantity']} | {row['delta']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |" for row in ci)
    cvt='\n'.join(f"| {row['model']} | {row['weighted_macro_f1']:.4f} | {row['weighted_mae']:.4f} |" for row in cvsummary)
    text=f'''# 问题2第二轮优化：回归、中性识别、分任务融合与增强权重

本轮已完成四项机制、全量训练内交叉验证和冻结后的官方验证，但未验证出相对于旧版的稳定提升。V2保留为实验候选；旧正式模型及原附件3预测保持默认，不将验证集中较好的消融重新指定为获胜模型。

## 1 实验边界

本轮是受到上一轮测试错误分析启发的探索性优化。未重新评价官方测试集，也没有使用其标签选参；不能将研究方向本身描述为完全未受旧测试影响。官方验证集已在上一轮使用，本轮只在参数冻结后报告对照，不把它称为新的独立测试，也不按本轮验证结果自动换模型。

使用全部 {frozen['train_n']} 条训练样本，按原始 video_id 做三折分组交叉验证。每折单独拟合词表、IDF、语义标准化和音视频标准化，冻结的通用预训练编码器不利用任何目标标签。完整输入及31个缺失情景共32个情景：30%比例覆盖7种模态组合和首、中、尾、随机位置，另含全模态随机10%、50%、70%。此设计不等于所有比例与位置的全因子穷举，也没有直接观察真实秒数。

预先固定完整输入权重0.4，其余31个情景共享0.6。分类最大化加权Macro-F1，回归最小化加权MAE，分别选择融合与增强权重。调参后的OOF分数是选模分数，未做嵌套交叉验证，不能作为无偏泛化性能。

## 2 完成的四项改进

1. **回归**：比较岭回归alpha=1/10/100与epsilon=0的绝对误差LinearSVR（C=0.1/1），并比较固定候选斜率0.8/1/1.2与偏移-0.1/0/0.1。SVR含正则化截距，与岭回归截距处理不同，比较是整体估计器比较，不能把所有差异归因于损失。所有强度裁剪至[-3,3]，所有迭代模型检查收敛。
2. **中性识别**：比较中性决策权重0.75/1/1.25/1.5。将原分类概率中性分量乘权重后归一化。输出是决策分数，不声称是经过概率校准的后验概率；没有强制修改回归符号。
3. **分任务融合**：分类与回归分别选择仅文本、音视频权重0.3的普通融合、权重0.3的可用比例融合、权重1的普通融合。两分支共享训练内拟合的基础特征映射，但可使用不同输入矩阵与维数。
4. **增强权重**：完整样本总权重候选0.25/0.6/1，其余质量平均分给三个局部缺失版本，每条原始样本总权重始终为1。权重1时增强样本不进入优化。分类与回归独立选择。

分类共 {frozen['classification_candidates']} 个含决策规则的候选，回归共 {frozen['regression_candidates']} 个含校准的候选。只使用单个固定增强种子17，以控制计算和便于比较；本轮未优化更强编码器或训练大型深度网络。

## 3 冻结选择

分类配置：`{json.dumps(cs['spec'],ensure_ascii=False)}`，中性权重 **{cs['neutral_weight']}**。

回归配置：`{json.dumps(rs['spec'],ensure_ascii=False)}`，强度变换 **clip({rs['scale']} × 原始输出 + ({rs['offset']}), -3, 3)**。

融合名称：text=仅文本；av03_plain=音视频权重0.3、无可用比例缩放；av03_quality=音视频权重0.3、有可用比例缩放；av1_plain=音视频权重1、无可用比例缩放。

| 训练内OOF（选模分数） | 加权Macro-F1 ↑ | 加权MAE ↓ |
|---|---:|---:|
{cvt}

参数选择与模型拟合分别保存哈希；首次官方验证之前即已冻结。旧协议也在各折重新拟合，用作同一分组划分的参照。

固定选定分类基础模型后，中性规则对比为：

| 中性权重 | 训练内加权Macro-F1 ↑ |
|---|---:|
{neutral_table}

中性规则保留1时，表示该候选搜索未支持调整阈值，不意味着中性识别问题已解决。回归家族各自最佳的训练内搜索分数如下；各家族可以选到不同融合和权重，此表不是只改变损失函数的严格消融：

| 回归家族 | 训练内加权MAE ↓ |
|---|---:|
{regression_table}

## 4 官方验证结果（728条，同一情景与权重）

| 模型 | 完整Macro-F1 ↑ | 完整MAE ↓ | 加权Macro-F1 ↑ | 加权MAE ↓ |
|---|---:|---:|---:|---:|
{table}

相对于旧单种子模型，完整输入中性召回率从 **{old['complete_neutral_recall']:.2%}** 到 **{new['complete_neutral_recall']:.2%}**；强情感（绝对强度≥2）MAE从 **{old['complete_strong_mae']:.4f}** 到 **{new['complete_strong_mae']:.4f}**；非中性分类与强度符号相反的样本从 **{old['complete_opposite_sign_count']:.0f}** 到 **{new['complete_opposite_sign_count']:.0f}**。这些分项可能与总体目标存在取舍，不假定四项修改都有效。

本轮主要发现：训练内候选排名在官方验证上未稳定复现；简单调高中性权重没有在OOF胜出。较弱正则化且不做0.8缩放时，强情感MAE可见改善，但整体MAE更差，说明“预测幅度变大”本身不等于整体更准确。默认中性规则被保留不代表中性识别已解决。保留增强的分类消融在验证上较好，只作为后续研究线索，不能据此绕过已冻结的选择协议。

消融以选定模型为起点，仅恢复指定模块到旧规则；其他参数保持不变，未给消融重新搜索最佳超参数。default_neutral_rule恢复中性权重1；no_regression_calibration恢复斜率1偏移0；ridge10_regression恢复岭回归alpha10并取消校准；shared_v1_fusion将两分支恢复为旧融合；old_augmentation_mass将完整样本权重恢复0.25。若冻结选项本来就是旧规则，对应消融重复是预期现象，不应声称该模块有效。

![验证集模块对照](validation_component_comparison.png)

## 5 不确定性

对官方验证集按原始视频聚类重采样500次，每次保留同一样本全部情景的配对关系。下表为V2减旧单种子模型，Macro-F1差值为正、MAE差值为负才有利。区间不包含调参过程和研究方向选择的不确定性，也没有进行多重比较校正。

| 指标 | 差值 | 95%分位区间 |
|---|---:|---:|
{cit}

区间跨0时不宣称显著改善。验证结果不理想的分项如实保留，不通过重复修改协议来掩盖。

## 6 交付与复现

附件3对齐版30条样本的新结果为 `附件3_问题2_优化版预测.csv`。附件3没有真实标签，不能判断该预测是否比旧版更准确。旧版模型、报告和预测保留在outputs/formal。两版本指标应在同一数据划分、同一缺失情景下比较，不将旧测试成绩与本轮验证成绩直接相减。

执行方法、模块替换接口、重载预测及校验见项目根目录 `REFINEMENT_README.md`。源码为refinement_model.py、refinement_run.py、refinement_predict.py、refinement_analysis.py。参数网格及验证边界见configs/refinement_protocol.json。完整局部日志、分折预测和搜索记录保存在本地。
'''
    (out/'问题2_第二轮优化报告.md').write_text(text,encoding='utf-8')
    print('Refinement report complete',json.dumps({'old':old,'new':new},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
