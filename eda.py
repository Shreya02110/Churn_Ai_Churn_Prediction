import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Professional Dark Theme ────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0F1117',
    'axes.facecolor':    '#0F1117',
    'axes.edgecolor':    '#2D2D3F',
    'axes.labelcolor':   '#E0E0E0',
    'xtick.color':       '#A0A0B0',
    'ytick.color':       '#A0A0B0',
    'text.color':        '#E0E0E0',
    'grid.color':        '#1E1E2E',
    'grid.linewidth':    0.5,
    'font.family':       'sans-serif',
    'font.size':         11,
})

COLORS = {
    'no':      '#00C897',
    'yes':     '#FF4C6A',
    'blue':    '#4E9AF1',
    'purple':  '#A78BFA',
    'orange':  '#FFA552',
    'teal':    '#00C897',
    'bg':      '#0F1117',
    'card':    '#1A1A2E',
    'text':    '#E0E0E0',
    'subtext': '#A0A0B0',
    'grid':    '#1E1E2E',
}

DATA_PATH = 'C:/Users/LENOVO/Desktop/Customer_Churn_Project/data/'

# ── Load Data ──────────────────────────────────────────
df = pd.read_csv(DATA_PATH + 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Dataset loaded! Shape: {df.shape}")
churn_rate = (df['Churn'] == 'Yes').mean() * 100
print(f"Overall Churn Rate: {churn_rate:.1f}%")

def save_fig(name):
    plt.savefig(DATA_PATH + name, dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()
    plt.close()
    print(f"  ✓ {name} saved!")

# ══════════════════════════════════════════════════════
# PLOT 1 — Churn Distribution (Donut)
# ══════════════════════════════════════════════════════
print("\n[Plot 1] Churn Distribution...")
counts = df['Churn'].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(COLORS['bg'])

wedges, texts, autotexts = ax.pie(
    counts.values,
    labels=None,
    autopct='%1.1f%%',
    startangle=90,
    colors=[COLORS['no'], COLORS['yes']],
    wedgeprops=dict(width=0.45, edgecolor=COLORS['bg'], linewidth=2),
    pctdistance=0.75,
)
for at in autotexts:
    at.set_color(COLORS['text'])
    at.set_fontsize(13)
    at.set_fontweight('bold')

ax.legend(
    wedges,
    [f"No Churn  ({counts['No']:,})", f"Churned  ({counts['Yes']:,})"],
    loc='lower center',
    bbox_to_anchor=(0.5, -0.08),
    ncol=2,
    fontsize=12,
    facecolor=COLORS['card'],
    edgecolor='none',
    labelcolor=COLORS['text'],
)
ax.set_title('Customer Churn Distribution',
             fontsize=16, fontweight='bold',
             color=COLORS['text'], pad=20)
centre = plt.Circle((0, 0), 0.55,
                     fc=COLORS['bg'])
ax.add_patch(centre)
ax.text(0, 0, f'{churn_rate:.1f}%\nChurn Rate',
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        color=COLORS['yes'])
save_fig('plot1_churn_distribution.png')

# ══════════════════════════════════════════════════════
# PLOT 2 — Contract Type vs Churn
# ══════════════════════════════════════════════════════
print("[Plot 2] Contract Type vs Churn...")
contract = df.groupby(['Contract', 'Churn']).size().unstack()
x      = np.arange(len(contract.index))
w      = 0.32
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(COLORS['bg'])

b1 = ax.bar(x - w/2, contract['No'],  w,
            color=COLORS['no'],  label='No Churn',
            zorder=3, edgecolor='none')
b2 = ax.bar(x + w/2, contract['Yes'], w,
            color=COLORS['yes'], label='Churned',
            zorder=3, edgecolor='none')

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 25,
            f'{int(bar.get_height()):,}',
            ha='center', va='bottom',
            fontsize=10, color=COLORS['text'])

churn_pcts = (contract['Yes'] / contract.sum(axis=1) * 100).round(1)
for i, (idx, pct) in enumerate(churn_pcts.items()):
    ax.text(i + w/2 + 0.02,
            contract.loc[idx, 'Yes'] / 2,
            f'{pct}%',
            ha='center', va='center',
            fontsize=9, color='white',
            fontweight='bold')

ax.set_title('Contract Type vs Churn',
             fontsize=16, fontweight='bold',
             color=COLORS['text'], pad=20)
ax.set_xticks(x)
ax.set_xticklabels(contract.index, fontsize=11)
ax.set_ylabel('Number of Customers',
              fontsize=12, color=COLORS['subtext'])
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.legend(fontsize=11, facecolor=COLORS['card'],
          edgecolor='none', labelcolor=COLORS['text'])
save_fig('plot2_contract_churn.png')

# ══════════════════════════════════════════════════════
# PLOT 3 — Tenure vs Monthly Charges (Scatter)
# ══════════════════════════════════════════════════════
print("[Plot 3] Tenure vs Monthly Charges...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(COLORS['bg'])

for label, color, zorder in [
    ('No',  COLORS['no'],  2),
    ('Yes', COLORS['yes'], 3)
]:
    sub = df[df['Churn'] == label]
    ax.scatter(sub['tenure'], sub['MonthlyCharges'],
               c=color, alpha=0.35, s=20,
               zorder=zorder,
               label='No Churn' if label == 'No' else 'Churned')

ax.set_title('Tenure vs Monthly Charges by Churn',
             fontsize=16, fontweight='bold',
             color=COLORS['text'], pad=20)
ax.set_xlabel('Tenure (Months)',
              fontsize=12, color=COLORS['subtext'])
ax.set_ylabel('Monthly Charges ($)',
              fontsize=12, color=COLORS['subtext'])
ax.yaxis.grid(True, zorder=0)
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.legend(fontsize=11, facecolor=COLORS['card'],
          edgecolor='none', labelcolor=COLORS['text'])

# Annotations
ax.annotate('High charges +\nshort tenure → churn',
            xy=(5, 95), xytext=(25, 85),
            fontsize=9, color=COLORS['yes'],
            arrowprops=dict(arrowstyle='->',
                           color=COLORS['yes'],
                           lw=1.2))
save_fig('plot3_tenure_charges.png')

# ══════════════════════════════════════════════════════
# PLOT 4 — Monthly Charges Distribution
# ══════════════════════════════════════════════════════
print("[Plot 4] Monthly Charges Distribution...")
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(COLORS['bg'])

for label, color in [('No', COLORS['no']), ('Yes', COLORS['yes'])]:
    sub = df[df['Churn'] == label]['MonthlyCharges']
    ax.hist(sub, bins=40, color=color, alpha=0.6,
            label='No Churn' if label == 'No' else 'Churned',
            edgecolor='none', zorder=3)
    ax.axvline(sub.mean(), color=color,
               linestyle='--', lw=1.5, alpha=0.9)
    ax.text(sub.mean() + 1, ax.get_ylim()[1] * 0.02,
            f'avg ${sub.mean():.0f}',
            color=color, fontsize=9)

ax.set_title('Monthly Charges Distribution by Churn',
             fontsize=16, fontweight='bold',
             color=COLORS['text'], pad=20)
ax.set_xlabel('Monthly Charges ($)',
              fontsize=12, color=COLORS['subtext'])
ax.set_ylabel('Number of Customers',
              fontsize=12, color=COLORS['subtext'])
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.legend(fontsize=11, facecolor=COLORS['card'],
          edgecolor='none', labelcolor=COLORS['text'])
save_fig('plot4_monthly_dist.png')

# ══════════════════════════════════════════════════════
# PLOT 5 — Internet Service vs Churn
# ══════════════════════════════════════════════════════
print("[Plot 5] Internet Service vs Churn...")
internet = df.groupby(['InternetService', 'Churn']).size().unstack()
x = np.arange(len(internet.index))
w = 0.32
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(COLORS['bg'])

b1 = ax.bar(x - w/2, internet['No'],  w,
            color=COLORS['no'],  label='No Churn',
            zorder=3, edgecolor='none')
b2 = ax.bar(x + w/2, internet['Yes'], w,
            color=COLORS['yes'], label='Churned',
            zorder=3, edgecolor='none')

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 15,
            f'{int(bar.get_height()):,}',
            ha='center', va='bottom',
            fontsize=10, color=COLORS['text'])

churn_pcts2 = (internet['Yes'] / internet.sum(axis=1) * 100).round(1)
for i, (idx, pct) in enumerate(churn_pcts2.items()):
    ax.text(i + w/2 + 0.02,
            internet.loc[idx, 'Yes'] / 2,
            f'{pct}%',
            ha='center', va='center',
            fontsize=9, color='white',
            fontweight='bold')

ax.set_title('Internet Service vs Churn',
             fontsize=16, fontweight='bold',
             color=COLORS['text'], pad=20)
ax.set_xticks(x)
ax.set_xticklabels(internet.index, fontsize=11)
ax.set_ylabel('Number of Customers',
              fontsize=12, color=COLORS['subtext'])
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.legend(fontsize=11, facecolor=COLORS['card'],
          edgecolor='none', labelcolor=COLORS['text'])
save_fig('plot5_internet_churn.png')

# ══════════════════════════════════════════════════════
# KEY INSIGHTS SUMMARY
# ══════════════════════════════════════════════════════
avg_tenure_churned  = df[df['Churn']=='Yes']['tenure'].mean()
avg_tenure_stay     = df[df['Churn']=='No']['tenure'].mean()
avg_charges_churned = df[df['Churn']=='Yes']['MonthlyCharges'].mean()
avg_charges_stay    = df[df['Churn']=='No']['MonthlyCharges'].mean()

print("\n" + "="*50)
print("          KEY INSIGHTS SUMMARY")
print("="*50)
print(f"  Total Customers       : {len(df):,}")
print(f"  Churn Rate            : {churn_rate:.1f}%")
print(f"  Avg tenure (churned)  : {avg_tenure_churned:.1f} months")
print(f"  Avg tenure (stayed)   : {avg_tenure_stay:.1f} months")
print(f"  Avg charges (churned) : ${avg_charges_churned:.2f}")
print(f"  Avg charges (stayed)  : ${avg_charges_stay:.2f}")
print("="*50)
print("\n  Contract type churn %:")
print(df.groupby('Contract')['Churn']
      .value_counts(normalize=True)
      .unstack().round(3).to_string())
 