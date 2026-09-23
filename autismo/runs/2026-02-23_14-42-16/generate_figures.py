"""
generate_figures.py
Generates two figures for the paper from original data files.

Required files in the same directory as this script:
  - hubs.json
  - fusiones.json
  - grafo.json

Usage:
  python generate_figures.py

Output (saved in same directory):
  - ttf_distribution.png
  - level_heatmap.png
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load files ────────────────────────────────────────────────────────────────

print("Loading files...")
try:
    with open(os.path.join(SCRIPT_DIR, 'hubs.json')) as f:
        hubs_data = json.load(f)
    with open(os.path.join(SCRIPT_DIR, 'fusiones.json')) as f:
        fus_data = json.load(f)
    with open(os.path.join(SCRIPT_DIR, 'grafo.json')) as f:
        grafo = json.load(f)
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Make sure hubs.json, fusiones.json and grafo.json are in the same directory as this script.")
    sys.exit(1)

hubs     = hubs_data['hubs']
fusiones = fus_data['fusiones']
nodos    = grafo['nodos']
node_idx = {n['id']: n for n in nodos}

print(f"  hubs.json     : {len(hubs)} HUBs")
print(f"  fusiones.json : {len(fusiones)} fusion events")
print(f"  grafo.json    : {len(nodos)} nodes")

# ── Verification ──────────────────────────────────────────────────────────────

ttfs  = [h['ftt_sum'] for h in hubs]
sizes = [len(h['clase_nodos']) for h in hubs]

print(f"\nVerification:")
print(f"  Total HUBs        : {len(hubs)}   (expected: 93)")
print(f"  TTF min           : {min(ttfs):.3f}  (expected: 1.885)")
print(f"  TTF max           : {max(ttfs):.3f}  (expected: 12.695)")
print(f"  Total fusions     : {len(fusiones)}  (expected: 1137)")
print(f"  HUB #1 class size : {sizes[0]}   (expected: 640)")

errors = []
if len(hubs) != 93:                errors.append(f"HUBs: {len(hubs)} != 93")
if abs(min(ttfs) - 1.885) > 0.001: errors.append(f"TTF min: {min(ttfs):.3f} != 1.885")
if len(fusiones) != 1137:          errors.append(f"Fusions: {len(fusiones)} != 1137")
if sizes[0] != 640:                errors.append(f"HUB #1 size: {sizes[0]} != 640")

if errors:
    print(f"\nWARNING: {len(errors)} discrepancy(ies) found:")
    for e in errors:
        print(f"  {e}")
else:
    print("  All checks passed")

# ── FIGURE 1: TTF distribution ────────────────────────────────────────────────

print("\nGenerating ttf_distribution.png...")

ttf_mega   = hubs[0]['ftt_sum']
size_mega  = sizes[0]
ttfs_rest  = [t for t, s in zip(ttfs, sizes) if s < 100]
sizes_rest = [s for t, s in zip(ttfs, sizes) if s < 100]

ttfs_sorted = sorted(ttfs)
q_bounds    = [ttfs_sorted[23], ttfs_sorted[46], ttfs_sorted[69]]
zone_colors = ['#4393c3', '#92c5de', '#f4a582', '#d6604d']
zone_labels = ['Q1', 'Q2', 'Q3', 'Q4']

def get_color(t):
    if t <= q_bounds[0]: return zone_colors[0]
    if t <= q_bounds[1]: return zone_colors[1]
    if t <= q_bounds[2]: return zone_colors[2]
    return zone_colors[3]

fig, ax = plt.subplots(figsize=(10, 5))

scatter_sizes = [max(20, s * 8) for s in sizes_rest]
point_colors  = [get_color(t) for t in ttfs_rest]

ax.scatter(ttfs_rest, [1] * len(ttfs_rest),
           s=scatter_sizes, c=point_colors, alpha=0.7, zorder=3)
ax.scatter([ttf_mega], [1], s=1200, c='#1a1a2e', marker='*', zorder=5)

boundaries = [1.8] + q_bounds + [13.2]
for qb, ql in zip(q_bounds, ['Q1/Q2', 'Q2/Q3', 'Q3/Q4']):
    ax.axvline(qb, color='gray', lw=0.8, ls='--', alpha=0.6)
    ax.text(qb + 0.05, 1.25, ql, fontsize=7, color='gray', ha='left')

for i in range(4):
    ax.axvspan(boundaries[i], boundaries[i+1], alpha=0.07, color=zone_colors[i])
    mid = (boundaries[i] + boundaries[i+1]) / 2
    ax.text(mid, 0.65, zone_labels[i], fontsize=9, ha='center',
            color=zone_colors[i], fontweight='bold')

ax.set_xlim(1.5, 13.2)
ax.set_ylim(0.4, 1.6)
ax.set_xlabel('TTF_sum (Total Topological Friction sum)', fontsize=11)
ax.set_yticks([])
ax.set_title(
    'TTF distribution across 93 HUBs\n'
    'Marker size proportional to equivalence class size. HUB #1 (\u2605) shown separately.',
    fontsize=10, pad=10
)

for sz, label in [(2, 'n=2'), (10, 'n=10'), (50, 'n=50')]:
    ax.scatter([], [], s=max(20, sz * 8), c='gray', alpha=0.5, label=label)
ax.scatter([], [], s=1200, c='#1a1a2e', marker='*', label=f'HUB #1 (n={size_mega})')
ax.legend(loc='upper right', fontsize=8, title='Class size', title_fontsize=8)

plt.tight_layout()
out1 = os.path.join(SCRIPT_DIR, 'ttf_distribution.png')
plt.savefig(out1, dpi=180, bbox_inches='tight')
plt.close()
print(f"  saved: {out1}")

# ── FIGURE 2: AC x NT level-pair heatmap ─────────────────────────────────────

print("Generating level_heatmap.png...")

level_pairs = defaultdict(int)
unresolved  = 0

for f in fusiones:
    a_id    = f['nodo_a_id']
    b_id    = f['nodo_b_id']
    a_actor = f['nodo_a_actor']
    ac_id, nt_id = (a_id, b_id) if a_actor == 'AutisticCognition' else (b_id, a_id)
    ac_lvl = node_idx.get(ac_id, {}).get('nivel')
    nt_lvl = node_idx.get(nt_id, {}).get('nivel')
    if ac_lvl and nt_lvl:
        level_pairs[(ac_lvl, nt_lvl)] += 1
    else:
        unresolved += 1

total_fus = sum(level_pairs.values())
top_pair  = max(level_pairs, key=level_pairs.get)
top_cnt   = level_pairs[top_pair]

print(f"  Resolved   : {total_fus}")
print(f"  Unresolved : {unresolved}")
print(f"  Most frequent pair : AC_L{top_pair[0]} x NT_L{top_pair[1]} "
      f"({top_cnt} events, {top_cnt/total_fus*100:.1f}%)")
if top_pair == (3, 6):
    print("  Matches paper: AC_L3 x NT_L6")
else:
    print(f"  Paper says AC_L3xNT_L6, data gives AC_L{top_pair[0]}xNT_L{top_pair[1]}")

matrix = np.zeros((7, 7))
for (al, nl), cnt in level_pairs.items():
    if 1 <= al <= 7 and 1 <= nl <= 7:
        matrix[al - 1][nl - 1] = cnt

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

for i in range(7):
    for j in range(7):
        val = int(matrix[i][j])
        pct = val / total_fus * 100
        if val > 0:
            color = 'white' if pct > 5 else 'black'
            txt   = f'{val}\n({pct:.1f}%)' if pct >= 2 else f'{val}'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7 if pct >= 2 else 6, color=color)

ax.set_xticks(range(7))
ax.set_yticks(range(7))
ax.set_xticklabels([f'NT L{i+1}' for i in range(7)], fontsize=9)
ax.set_yticklabels([f'AC L{i+1}' for i in range(7)], fontsize=9)
ax.set_xlabel('NT node level', fontsize=11)
ax.set_ylabel('AC node level', fontsize=11)

rect = plt.Rectangle(
    (top_pair[1] - 1 - 0.5, top_pair[0] - 1 - 0.5), 1, 1,
    fill=False, edgecolor='blue', lw=2.5
)
ax.add_patch(rect)

plt.colorbar(im, ax=ax, label='Fusion event count')
ax.set_title(
    f'AC x NT level-pair heatmap across {total_fus} fusion events\n'
    f'Blue box: AC_L{top_pair[0]} x NT_L{top_pair[1]} '
    f'(most frequent pair, {top_cnt} events, {top_cnt/total_fus*100:.1f}%)',
    fontsize=10, pad=10
)

plt.tight_layout()
out2 = os.path.join(SCRIPT_DIR, 'level_heatmap.png')
plt.savefig(out2, dpi=180, bbox_inches='tight')
plt.close()
print(f"  saved: {out2}")

print("\nDone.")