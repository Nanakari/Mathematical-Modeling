"""Evaluation figures, error tables, readable report and compact run package."""
import csv
import json
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def table(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def make_report(out,report,metrics,predictions,valid):
    winner=report['selected_experiment']
    clean=[p for p in predictions if p['experiment']==winner and p['scenario']=='complete']
    errors=[]
    counts={valid[i]['sample_id']:{m:int(valid[i][m+'_observed'].sum()) for m in ('text','audio','vision')} for i in range(len(valid))}
    for p in clean:
        errors.append({**p,'absolute_error':abs(p['pred_intensity']-p['true_intensity']),
                       'classification_wrong':p['pred_class']!=p['true_class'],
                       'class_regression_disagree':p['pred_class']!=int(np.sign(p['pred_intensity'])+1),
                       **{m+'_observed_positions':v for m,v in counts[p['sample_id']].items()}})
    table(out/'error_cases.csv',sorted(errors,key=lambda p:p['absolute_error'],reverse=True)[:20])
    groups=[]
    for c in range(3):
        selected=[p for p in errors if p['true_class']==c]
        groups.append({'true_class':c,'n':len(selected),
                       'mae':float(np.mean([p['absolute_error'] for p in selected])) if selected else None,
                       'classification_error_rate':float(np.mean([p['classification_wrong'] for p in selected])) if selected else None})
    table(out/'error_by_class.csv',groups)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for exp in report['experiments']:
        h=json.loads((out/exp['experiment']/'history.json').read_text(encoding='utf-8'))
        axes[0].plot(range(1,len(h)+1),[r['selection_score'] for r in h],marker='.',label=exp['experiment'])
    axes[0].set(xlabel='Epoch',ylabel='Validation selection score (lower is better)',title='Checkpoint selection')
    axes[0].legend(fontsize=8)
    matrix=confusion_matrix([p['true_class'] for p in clean],[p['pred_class'] for p in clean],labels=[0,1,2])
    axes[1].imshow(matrix,cmap='Blues')
    for i in range(3):
        for j in range(3):axes[1].text(j,i,str(matrix[i,j]),ha='center',va='center')
    axes[1].set(xticks=[0,1,2],yticks=[0,1,2],xlabel='Predicted class',ylabel='True class',title='Selected model: complete input')
    axes[2].scatter([p['true_intensity'] for p in clean],[p['pred_intensity'] for p in clean],s=14,alpha=.65)
    axes[2].plot([-3,3],[-3,3],'k--',lw=1)
    axes[2].set(xlabel='True intensity',ylabel='Predicted intensity',xlim=(-3,3),ylim=(-3,3),title='Selected model: regression')
    fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for exp in report['experiments']:
        name=exp['experiment'];data=[r for r in metrics if r['experiment']==name]
        complete=next(r for r in data if r['scenario']=='complete')
        curve=[r for r in data if '|random|text+audio+vision|' in r['scenario']]
        rates=sorted({float(r['scenario'].split('|')[-1]) for r in curve})
        for ax,key in zip(axes,['mae','f1_macro']):
            values=[complete[key]]+[np.mean([r[key] for r in curve if float(r['scenario'].split('|')[-1])==rate]) for rate in rates]
            ax.plot([0]+rates,values,marker='o',label=name)
            ax.set(xlabel='Requested missing ratio',ylabel=key,title='All modalities, random continuous intervals')
            ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.savefig(out/'missingness_curves.png',dpi=150);plt.close(fig)
    combinations=['text','audio','vision','text+audio','text+vision','audio+vision','text+audio+vision']
    positions=['start','middle','end','random']
    heat=np.zeros((len(combinations),len(positions)))
    for i,combo in enumerate(combinations):
        for j,pos in enumerate(positions):
            values=[r['mae'] for r in metrics if r['experiment']==winner and r['scenario']!='complete'
                    and r['scenario'].split('|')[1:]==[pos,combo,'0.3']]
            heat[i,j]=np.mean(values) if values else np.nan
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    im=ax.imshow(heat,cmap='YlOrRd',aspect='auto')
    ax.set(xticks=range(4),xticklabels=positions,yticks=range(7),yticklabels=combinations,
           title='Selected model: modality and position at 30% missingness')
    for i in range(7):
        for j in range(4):ax.text(j,i,f'{heat[i,j]:.3f}',ha='center',va='center')
    fig.colorbar(im,ax=ax,label='MAE')
    fig.savefig(out/'position_modality_heatmap.png',dpi=150);plt.close(fig)
    lines=['# 问题 2 小规模完整流程结果','',
           f"使用官方训练集中的 {report['train_n']} 条样本和验证集中的 {report['valid_n']} 条样本。标准化仅拟合这批训练样本，未评价官方测试集。",'',
           f"依据预先配置的验证评分选出的模型是 **{winner}**。评分为完整输入与指定缺失情景下 MAE + (1 − Macro-F1) 的平均值；权重可替换。",'',
           '| 设置 | 最佳轮次 | 选择评分 | 完整输入 Accuracy | Macro-F1 | MAE | Pearson |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for e in report['experiments']:
        r=next(x for x in metrics if x['experiment']==e['experiment'] and x['scenario']=='complete')
        pearson=f"{r['pearson']:.4f}" if r['pearson'] is not None else '未定义'
        lines.append(f"| {e['experiment']} | {e['best_epoch']} | {e['selection_score']:.4f} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | {r['mae']:.4f} | {pearson} |")
    if 'constant_reference' in report:
        r=report['constant_reference']['metrics']
        lines+=['',f"作为参考，使用训练集多数类和训练集强度中位数作恒定预测，在同一验证子集上的 Accuracy={r['accuracy']:.4f}、Macro-F1={r['f1_macro']:.4f}、MAE={r['mae']:.4f}。常数回归输出的 Pearson 无定义。此参考不参与选模。"]
    lines+=['','## 结果与解释边界','',
            f"每个模型均在同一份 {report['scenario_library']['scenario_count']} 个验证情景上评价。metrics.csv 保存完整指标，validation_predictions.csv 保存逐样本预测。缺失区间、实际缺失比例和种子保存在压缩场景库中。",'',
            'clean_concat 与 augmented_concat 的差别是缺失增强；augmented_concat 与 augmented_gated 的差别是融合结构及其参数量。三者共用样本、标准化器、编码器配置和验证区间。', '',
            '验证集同时用于选轮次、选模型与诊断，所以这些结果不是独立留出集上的无偏性能估计。只有一个训练种子，不能据此声称显著提高。当前文本编码器为随机初始化的词元嵌入，尚未采用已确认词表的预训练 BERT。', '',
            'error_cases.csv 按强度绝对误差列出前 20 条样本，并记录分类错误、分类与回归符号不一致、原始模态可用位置数。error_by_class.csv 给出分类别误差。这些是错误定位线索，不是因果归因。', '',
            f"附件 3 aligned 的 {report['special_predictions']} 条样本已生成预测。attachment3_DRAFT_predictions.csv 使用文件名追踪键及暂定列名；无标签，不能据其计算性能。正式提交格式与文本编码器仍需确定。",'',
            '## 文件说明','',
            '- selected_model.pt：选中模型的推理参数，不含优化器。',
            '- 各实验 best.pt / last.pt：最佳推理参数 / 最后一轮模型及优化器状态。',
            '- standardizer.json：训练集标准化参数。',
            '- config.json / split_ids.json / report.json：配置、样本划分记录、环境与运行摘要。',
            '- diagnostics.png / missingness_curves.png / position_modality_heatmap.png：选模、混淆矩阵、回归散点、缺失性能曲线及模态与位置对比。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def package(root,out):
    target=out/'preliminary_reproduction.zip'
    files=list(root.glob('*.py'))+[root/'requirements.txt',root/'requirements-torch.txt']
    files += list((root/'configs').glob('*.json'))+[root/'docs'/'完整流程与替换接口.md']
    files += [out/name for name in ['selected_model.pt','standardizer.json','config.json','split_ids.json','report.json',
                                   'RESULTS.md','metrics.csv','attachment3_DRAFT_predictions.csv','diagnostics.png','missingness_curves.png','position_modality_heatmap.png']]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in files:
            if not file.exists():raise FileNotFoundError(file)
            z.write(file,Path('problem2')/file.relative_to(root))
    if target.stat().st_size>50_000_000:raise ValueError('Package exceeds 50 MB')
    print(f'Preliminary package: {target.name}, {target.stat().st_size/1e6:.2f} MB; not formal submission')
