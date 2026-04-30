import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path('figures').mkdir(exist_ok=True)
df = pd.read_csv('results/kv_compression_summary.csv')

# Plot A: Accuracy vs compression ratio
uni = df[df['compression_ratio'].notna()]
datasets = ['docvqa', 'mmmu', 'realworldqa', 'mathvista']
methods = sorted([m for m in uni['method'].unique() if m != 'modality'])
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
for ax, ds in zip(axes, datasets):
    sub = uni[uni['dataset'] == ds]
    for m in methods:
        sm = sub[sub['method'] == m].sort_values('compression_ratio')
        if len(sm) == 0:
            continue
        ax.plot(sm['compression_ratio'], sm['accuracy'], marker='o', label=m, linewidth=2)
    ax.set_title(ds)
    ax.set_xlabel('Compression ratio')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Accuracy')
axes[0].legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig1_accuracy_vs_compression.pdf', bbox_inches='tight')
plt.savefig('figures/fig1_accuracy_vs_compression.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1')

# Plot B: Modality vs uniform h2o
mod = df[df['method'] == 'modality'].copy()
unih2o = df[df['method'] == 'h2o'][['dataset', 'compression_ratio', 'accuracy']].rename(
    columns={'accuracy': 'h2o_acc', 'compression_ratio': 'matched_ratio'})
mod['matched_ratio'] = (mod['mean_effective_compression'] * 10).round() / 10
mod['matched_ratio'] = mod['matched_ratio'].clip(upper=0.7)
paired = mod.merge(unih2o, on=['dataset', 'matched_ratio'], how='left')

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(paired))
ax.bar(x - 0.2, paired['h2o_acc'], 0.4, label='Uniform h2o', color='#888')
ax.bar(x + 0.2, paired['accuracy'], 0.4, label='Modality-h2o', color='#3a7')
labels = paired.apply(lambda r: r['dataset'] + '\ni' + str(r['image_ratio']) + ' t' + str(r['text_ratio']), axis=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Accuracy')
ax.set_title('Modality-aware vs uniform compression')
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures/fig2_modality_vs_uniform.pdf', bbox_inches='tight')
plt.savefig('figures/fig2_modality_vs_uniform.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2')

# Plot C: Prefill speedup heatmap
prefill = uni[uni['method'] != 'modality'].pivot_table(
    index=['dataset', 'method'], columns='compression_ratio', values='mean_prefill_ms')
baseline = prefill[0.0]
speedup = (baseline.values[:, None] - prefill.values) / baseline.values[:, None] * 100
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(speedup, aspect='auto', cmap='RdYlGn', vmin=-30, vmax=60)
ax.set_xticks(range(prefill.shape[1]))
ax.set_xticklabels([str(c) for c in prefill.columns])
ax.set_yticks(range(prefill.shape[0]))
ax.set_yticklabels([a + '/' + b for a, b in prefill.index], fontsize=8)
for i in range(prefill.shape[0]):
    for j in range(prefill.shape[1]):
        v = speedup[i, j]
        ax.text(j, i, 'NaN' if np.isnan(v) else str(round(v, 1)), ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, label='Prefill speedup (%)')
ax.set_title('Prefill time reduction vs. baseline')
ax.set_xlabel('Compression ratio')
plt.tight_layout()
plt.savefig('figures/fig3_prefill_speedup.pdf', bbox_inches='tight')
plt.savefig('figures/fig3_prefill_speedup.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3')

# Plot D: Memory savings heatmap
mem = uni[uni['method'] != 'modality'].pivot_table(
    index=['dataset', 'method'], columns='compression_ratio', values='peak_gpu_mem_gb')
baseline = mem[0.0]
saved = (baseline.values[:, None] - mem.values) / baseline.values[:, None] * 100
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(saved, aspect='auto', cmap='Blues', vmin=0, vmax=10)
ax.set_xticks(range(mem.shape[1]))
ax.set_xticklabels([str(c) for c in mem.columns])
ax.set_yticks(range(mem.shape[0]))
ax.set_yticklabels([a + '/' + b for a, b in mem.index], fontsize=8)
for i in range(mem.shape[0]):
    for j in range(mem.shape[1]):
        v = saved[i, j]
        ax.text(j, i, 'NaN' if np.isnan(v) else str(round(v, 2)), ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax, label='Memory saved (%)')
ax.set_title('Peak GPU memory reduction vs. baseline')
ax.set_xlabel('Compression ratio')
plt.tight_layout()
plt.savefig('figures/fig4_memory_savings.pdf', bbox_inches='tight')
plt.savefig('figures/fig4_memory_savings.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4')

print('All figures written to figures/')
