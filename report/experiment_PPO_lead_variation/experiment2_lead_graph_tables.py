
#!/usr/bin/env python3
"""Reproduce Experiment 2 lead-time sensitivity tables, charts, and concise report.

Usage:
  python experiment2_report.py --input-dir evaluation/results --output-dir evaluation/experiment2_report

Requirements: pandas, numpy, matplotlib, python-docx.
Reads lead_sensitivity_*missions_eval_lead*.csv; no training or simulator calls.
Outputs: PDF (if LibreOffice available), DOCX, PNG charts, summary CSV tables.
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

TEAL = '#187F78'
ORANGE = '#D18A40'
PURPLE = '#7854A8'
GRAY = '#74818C'
DARK = '#203041'
LIGHT = '#EEF3F5'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, default=Path('evaluation/results'))
    p.add_argument('--output-dir', type=Path, default=Path('evaluation/experiment2_report'))
    p.add_argument('--missions', type=int, default=50)
    p.add_argument('--reference-lead', type=float, default=5)
    return p.parse_args()

def policy_name(name):
    if name == 'margin rule': return 'Margin rule'
    if name == 'always WAIT': return 'Always WAIT'
    m = re.search(r'trigger_ppo_lead([\d.]+)_seed(\d+)', name)
    return f'PPO L={float(m.group(1)):g}' if m else name

def read_files(folder, missions):
    files = sorted(folder.glob(f'lead_sensitivity_{missions}missions_eval_lead*.csv'))
    if not files:
        raise FileNotFoundError(f'No lead-sensitivity CSVs in {folder}.')
    frames = []
    for file in files:
        mt = re.search(r'_eval_lead([\d.]+)\.csv$', file.name)
        if not mt: continue
        d = pd.read_csv(file)
        required = {'policy','precision','recall','timely recall','false-alarm rate','lead time median',
                    'monitored tasks','monitored misses','excluded: no report before deadline',
                    'of which missed','excluded: outcome unknown'}
        missing = required - set(d.columns)
        if missing: raise ValueError(f'{file}: missing {sorted(missing)}')
        d['eval_threshold'] = float(mt.group(1))
        d['policy_short'] = d['policy'].map(policy_name)
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(['eval_threshold','policy_short']).any():
        raise ValueError('Duplicate policy entries for an evaluation threshold.')
    fixed = ['precision','recall','false-alarm rate','lead time median','lead time p10',
             'monitored tasks','monitored misses','excluded: no report before deadline','of which missed']
    for name, grp in data.groupby('policy_short'):
        for col in fixed:
            if col in data and grp[col].nunique() > 1:
                raise ValueError(f'{name}: {col} varies by evaluation threshold; check mission reproducibility.')
    return data

def derive(data, reference):
    thresholds = sorted(data['eval_threshold'].unique())
    if reference not in thresholds:
        raise ValueError(f'Reference lead {reference:g} not found in {thresholds}')
    core = data[data.eval_threshold == reference].copy()
    n = int(core['monitored tasks'].iloc[0]); misses = int(core['monitored misses'].iloc[0]); negatives=n-misses
    core['TP'] = np.rint(core['recall'] * misses).astype(int)
    core['FN'] = misses - core['TP']
    core['FP'] = np.rint(core['false-alarm rate'] * negatives).astype(int)
    core['TN'] = negatives - core['FP']
    core['timely_TP'] = np.rint(core['timely recall'] * misses).astype(int)
    core['alerts'] = core.TP + core.FP
    core['actionable_all_misses'] = core.timely_TP / (misses + int(core['of which missed'].iloc[0]))
    core['policy_order'] = core.policy_short.map(lambda s: float(s.split('=')[1]) if s.startswith('PPO L=') else -1)
    core = core.sort_values('policy_order')
    return core, n, misses, negatives

def clean_plot():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':DARK,
                         'text.color':DARK,'axes.edgecolor':'#CAD5DB',
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.22,'figure.facecolor':'white',
                         'axes.facecolor':'white','savefig.facecolor':'white'})

def plot_tradeoff(core, out):
    fig,ax=plt.subplots(figsize=(8.0,4.2))
    ppo=core[core.policy_short.str.startswith('PPO')]
    for _,r in ppo.iterrows():
        ax.scatter(r['FP'],r['timely_TP'],s=115,c=TEAL,zorder=3)
        ax.annotate(r.policy_short.replace('PPO ',''), (r['FP'],r['timely_TP']),
                    xytext=(5,7),textcoords='offset points',fontsize=9)
    m=core[core.policy_short=='Margin rule'].iloc[0]
    ax.scatter(m.FP,m.timely_TP,marker='D',s=115,c=ORANGE,zorder=4,label='Margin rule')
    ax.annotate('Margin rule',(m.FP,m.timely_TP),xytext=(-66,-17),textcoords='offset points',fontsize=9)
    ax.set_xlabel('False alerts (lower is better)')
    ax.set_ylabel('Timely correct alerts (higher is better)')
    ax.set_title('Actionable-alert tradeoff at evaluation lead = 5',fontweight='bold',loc='left',pad=14)
    ax.set_xlim(16,45);ax.set_ylim(169,190)
    ax.text(.015,.035,'Upper-left is preferable',transform=ax.transAxes,color=GRAY,fontsize=9)
    fig.tight_layout();fig.savefig(out,dpi=190,bbox_inches='tight');plt.close(fig)

def plot_horizon(data,out):
    fig,ax=plt.subplots(figsize=(8.0,4.3))
    chosen={'Margin rule':(ORANGE,'--',2),'PPO L=2':(TEAL,'-',2),
            'PPO L=5':(PURPLE,'-',2),'PPO L=15':('#53627E','-',2)}
    for name,(col,ls,lw) in chosen.items():
        d=data[data.policy_short==name].sort_values('eval_threshold')
        ax.plot(d.eval_threshold,d['timely recall']*100,marker='o',markersize=4,
                linewidth=lw,linestyle=ls,label=name,color=col)
    ax.set_xlabel('Required warning lead time (TA-RWARE time units)')
    ax.set_ylabel('Actionable recall (%)')
    ax.set_title('Warning effectiveness decreases sharply at longer horizons',fontweight='bold',loc='left',pad=14)
    ax.set_xticks(sorted(data.eval_threshold.unique()))
    ax.set_ylim(0,100);ax.legend(frameon=False,ncol=2,loc='upper right')
    fig.tight_layout();fig.savefig(out,dpi=190,bbox_inches='tight');plt.close(fig)

def plot_coverage(core,out):
    n=int(core['monitored tasks'].iloc[0]); excluded=int(core['excluded: no report before deadline'].iloc[0]);
    misses=int(core['monitored misses'].iloc[0]); excluded_misses=int(core['of which missed'].iloc[0]);
    total_misses=misses+excluded_misses
    fig,ax=plt.subplots(figsize=(8.0,1.65))
    ax.barh(0,misses,color=TEAL,height=.42,label=f'Monitorable missed tasks: {misses}')
    ax.barh(0,excluded_misses,left=misses,color=ORANGE,height=.42,label=f'No pre-deadline report: {excluded_misses}')
    ax.set_xlim(0,total_misses);ax.set_yticks([]);ax.set_xticks([])
    for x,t in [(misses/2,f'{misses} / {total_misses}'),(misses+excluded_misses/2,f'{excluded_misses} / {total_misses}')]:
        ax.text(x,0,t,ha='center',va='center',color='white',fontweight='bold',fontsize=11)
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,-.39),ncol=2,fontsize=9)
    ax.set_title('Observability of deadline misses',fontweight='bold',loc='left',pad=9)
    fig.tight_layout();fig.savefig(out,dpi=190,bbox_inches='tight');plt.close(fig)

def set_cell_shading(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);tcPr.append(shd)

def style_doc(doc):
    s=doc.styles['Normal'];s.font.name='Aptos';s.font.size=Pt(9);s.font.color.rgb=RGBColor(39,51,63)
    s.paragraph_format.space_after=Pt(5)
    for nm,size in [('Title',20),('Heading 1',12),('Heading 2',10)]:
        z=doc.styles[nm];z.font.name='Aptos';z.font.size=Pt(size);z.font.bold=True;z.font.color.rgb=RGBColor(24,85,83)
    section=doc.sections[0];section.top_margin=Inches(.73);section.bottom_margin=Inches(.70)
    section.left_margin=Inches(.83);section.right_margin=Inches(.83)
    sec_footer=section.footer.paragraphs[0];sec_footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    sec_footer.text='Experiment 2  |  TA-RWARE trigger evaluation'
    sec_footer.style='Caption'

def add_table(doc, headers, rows, widths=None):
    table=doc.add_table(rows=1,cols=len(headers));table.alignment=WD_TABLE_ALIGNMENT.CENTER
    table.style='Table Grid'
    for i,h in enumerate(headers):
        c=table.rows[0].cells[i];c.text=str(h);set_cell_shading(c,'E2ECEC')
        for p in c.paragraphs:
            for run in p.runs: run.bold=True;run.font.size=Pt(8)
    for ri,row in enumerate(rows):
        cells=table.add_row().cells
        for j,v in enumerate(row):
            cells[j].text=str(v)
            if ri%2: set_cell_shading(cells[j],'F6F9F9')
            cells[j].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[j].paragraphs:
                for run in p.runs:run.font.size=Pt(8)
    doc.add_paragraph()

def add_figure(doc,path,caption,width=6.55):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(path),width=Inches(width))
    cp=doc.add_paragraph(caption);cp.style='Caption';cp.alignment=WD_ALIGN_PARAGRAPH.CENTER

def write_report(core,data,n,misses,negative,out,figs,reference):
    doc=Document();style_doc(doc)
    title=doc.add_paragraph(style='Title');title.add_run('Experiment 2 | Lead-Time Sensitivity')
    sub=doc.add_paragraph('TA-RWARE sequential WAIT / ALERT policy - concise experimental report')
    sub.style='Subtitle'
    doc.add_heading('Objective and protocol',level=1)
    doc.add_paragraph('We examine how the training reward saturation horizon (LEAD_MIN) affects the behavior of PPO task-level early-warning policies. Seven PPO policies were trained with LEAD_MIN = 2, 3, 5, 8, 10, 15, and 20; their other PPO settings and the original terminal reward structure were held constant. Each policy was evaluated deterministically on the same 50 held-out mission IDs, alongside an always-WAIT policy and a deadline-margin rule. Evaluation warning thresholds are common to all policies.')
    doc.add_paragraph(f'The evaluation comprises {n:,} monitorable tasks, including {misses} deadline misses ({100*misses/n:.1f}%). The reference comparison uses an actionable-warning requirement of {reference:g} TA-RWARE time units before the deadline. Results reflect one training seed per PPO configuration; no uncertainty intervals can be inferred from these aggregates.')
    doc.add_heading('Key performance comparison',level=1)
    picks=['Margin rule','PPO L=2','PPO L=5','PPO L=15','PPO L=20']
    c=core.set_index('policy_short')
    rows=[]
    for k in picks:
        r=c.loc[k];rows.append([k,f'{r.precision*100:.1f}',f'{r.recall*100:.1f}',f'{r["timely recall"]*100:.1f}',
                                f'{r["false-alarm rate"]*100:.2f}',f'{int(r.FP)} / {int(r.FN)}'])
    add_table(doc,['Policy','Precision %','Recall %','Timely recall %','FAR %','FP / FN'],rows)
    doc.add_paragraph('Among the selected policies, PPO L=15 yields the lowest false-alarm rate (1.29%; 21 false alarms) and 60.5% actionable recall, but misses six monitorable deadline failures. PPO L=5 provides the highest actionable recall at the five-unit requirement (62.5%; 185 timely alerts), with 38 false alarms. PPO L=2 is a balanced alternative: 22 false alerts and only three missed alerts, compared with 40 and zero for the margin rule.')
    add_figure(doc,figs['tradeoff'],'Figure 1. Counts of correct alerts with at least five time units of lead versus false alarms. Each point represents a distinct trained policy.',6.5)
    doc.add_heading('Lead-time robustness',level=1)
    add_figure(doc,figs['horizon'],'Figure 2. Actionable recall evaluated at several common warning horizons. Training LEAD_MIN and evaluation horizon are different parameters.',5.85)
    doc.add_paragraph('Across PPO variants, median true-positive warning lead time is only 5-6 units and tenth-percentile lead time is 2 units. At a requirement of 20 units, actionable recall for all PPO policies is at most about 0.7%. Thus, increasing training LEAD_MIN does not reliably shift warnings earlier; the main observed variation is in false-alarm behavior. Whether this stems from report timing, limited early predictive information, or the learned policies cannot be established from the aggregated CSVs alone.')
    doc.add_heading('Observability limit and interpretation',level=1)
    add_figure(doc,figs['coverage'],'Figure 3. Of 627 deadline-missing tasks, 331 have no report before their deadline and are excluded from policy evaluation.',5.9)
    coverage=misses/(misses+int(core['of which missed'].iloc[0]))
    timely5=int(c.loc['PPO L=5','timely_TP']);totalmiss=misses+int(core['of which missed'].iloc[0])
    doc.add_paragraph(f'Conditional recall is calculated over the {misses} monitorable deadline misses. Counting all {totalmiss} deadline misses, maximum detectable recall is {coverage*100:.1f}%, even for an ideal trigger over the existing observation stream. For PPO L=5, its {timely5} actionable alerts correspond to {100*timely5/totalmiss:.1f}% of all deadline misses, versus 62.5% of monitorable misses. The excluded cases should be treated as a monitoring-availability limitation, not PPO classification errors.')
    doc.add_heading('Conclusions and next checks',level=1)
    doc.add_paragraph('The lead-time sweep reveals a precision-versus-coverage tradeoff, rather than a consistent gain in early alerting. PPO often reduces false alarms relative to the margin rule while retaining near-complete conditional recall. The apparent advantages are preliminary because training was not repeated across seeds and the analysis contains only aggregate policy metrics.')
    doc.add_paragraph('Next, measure earliest report lead time per task, produce paired per-task alert-time comparisons, and rerun selected configurations across multiple independent seeds. Confirm that mission generation is deterministic for each held-out mission ID. A model should not be selected solely from the highest mean reward: the reward changes across training configurations and imposes an application-specific tradeoff.')
    doc.add_paragraph('Source: seven evaluation CSVs named lead_sensitivity_50missions_eval_lead{2,3,5,8,10,15,20}.csv. FP/FN counts were reconstructed from reported rates and shared class totals, with rounding to the nearest integer.')
    doc.save(out)

def main():
    a=parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    data=read_files(a.input_dir,a.missions)
    core,n,misses,negatives=derive(data,a.reference_lead)
    # Reproducible outputs: save all policy summaries and threshold curves
    core.drop(columns=['policy_order']).to_csv(a.output_dir/'table_policy_comparison.csv',index=False)
    data[['policy_short','eval_threshold','timely recall','mean reward per task']].to_csv(a.output_dir/'table_horizon_sensitivity.csv',index=False)
    clean_plot()
    figs={x:a.output_dir/f'figure_{x}.png' for x in ['tradeoff','horizon','coverage']}
    plot_tradeoff(core,figs['tradeoff']);plot_horizon(data,figs['horizon']);plot_coverage(core,figs['coverage'])
    docx=a.output_dir/'Experiment2_Lead_Time_Sensitivity_Report.docx'
    write_report(core,data,n,misses,negatives,docx,figs,a.reference_lead)
    if shutil.which('libreoffice'):
        subprocess.run(['libreoffice','-env:UserInstallation=file:///tmp/lo_exp2_report',
                        '--headless','--convert-to','pdf','--outdir',str(a.output_dir),str(docx)],
                       check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    print('Report and reproducing figures/tables saved in:',a.output_dir.resolve())
    for f in sorted(a.output_dir.iterdir()): print(' ',f.name)

if __name__=='__main__':main()
