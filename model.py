import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────
data_path  = 'C:/Users/LENOVO/Desktop/Customer_Churn_Project/data/'
model_path = 'C:/Users/LENOVO/Desktop/Customer_Churn_Project/models/'

# ── Theme ──────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0F1117',
    'axes.facecolor':   '#0F1117',
    'axes.edgecolor':   '#2D2D3F',
    'axes.labelcolor':  '#E0E0E0',
    'xtick.color':      '#A0A0B0',
    'ytick.color':      '#A0A0B0',
    'text.color':       '#E0E0E0',
    'grid.color':       '#1E1E2E',
    'grid.linewidth':   0.5,
    'font.family':      'sans-serif',
    'font.size':        11,
})

COLORS = {
    'no':      '#00C897',
    'yes':     '#FF4C6A',
    'blue':    '#4E9AF1',
    'purple':  '#A78BFA',
    'orange':  '#FFA552',
    'bg':      '#0F1117',
    'card':    '#1A1A2E',
    'text':    '#E0E0E0',
    'subtext': '#A0A0B0',
}

# ══════════════════════════════════════════════════════
# STEP 1 — Load Processed Data
# ══════════════════════════════════════════════════════
X_train = pd.read_csv(data_path + 'X_train.csv')
X_test  = pd.read_csv(data_path + 'X_test.csv')
y_train = pd.read_csv(data_path + 'y_train.csv').values.ravel()
y_test  = pd.read_csv(data_path + 'y_test.csv').values.ravel()

print("Data loaded!")
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# ══════════════════════════════════════════════════════
# STEP 2 — SMOTE + Train Models
# ══════════════════════════════════════════════════════
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE — Train size : {X_train_sm.shape}")
print(f"Churn ratio after SMOTE  : {y_train_sm.mean()*100:.1f}%")

models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=1000, C=0.1),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1),
    'XGBoost': XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,
        random_state=42,
        eval_metric='logloss',
        verbosity=0),
}

results = {}
print("\n=== Training Models ===")
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    roc    = roc_auc_score(y_test, y_prob)
    results[name] = {
        'model':  model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'acc':    acc,
        'roc':    roc,
    }
    print(f"  {name:25s} | Acc: {acc*100:.2f}% | ROC-AUC: {roc:.4f}")

best_name = max(results, key=lambda x: results[x]['acc']) 
print(f"\n  Best Model: {best_name}")

# ══════════════════════════════════════════════════════
# STEP 3 — Model Comparison Plot
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(COLORS['bg'])

names      = list(results.keys())
accs       = [results[n]['acc']*100 for n in names]
roc_vals   = [results[n]['roc']*100 for n in names]
bar_colors = [COLORS['blue'], COLORS['purple'], COLORS['orange']]

ax = axes[0]
bars = ax.bar(names, accs, color=bar_colors, width=0.45, zorder=3)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{val:.2f}%',
            ha='center', fontsize=10,
            color=COLORS['text'], fontweight='bold')
ax.set_title('Model Accuracy Comparison',
             fontsize=14, fontweight='bold',
             color=COLORS['text'], pad=15)
ax.set_ylabel('Accuracy (%)', fontsize=11, color=COLORS['subtext'])
ax.set_ylim(60, 100)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.tick_params(axis='x', labelsize=9)

ax = axes[1]
bars = ax.bar(names, roc_vals, color=bar_colors, width=0.45, zorder=3)
for bar, val in zip(bars, roc_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{val:.2f}%',
            ha='center', fontsize=10,
            color=COLORS['text'], fontweight='bold')
ax.set_title('Model ROC-AUC Comparison',
             fontsize=14, fontweight='bold',
             color=COLORS['text'], pad=15)
ax.set_ylabel('ROC-AUC (%)', fontsize=11, color=COLORS['subtext'])
ax.set_ylim(60, 100)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.savefig(data_path + 'plot6_model_comparison.png',
            dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.show()
print("Plot 6 saved!")

# ══════════════════════════════════════════════════════
# STEP 4 — Confusion Matrix (Best Model)
# ══════════════════════════════════════════════════════
best = results[best_name]
cm   = confusion_matrix(y_test, best['y_pred'])

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(COLORS['bg'])
sns.heatmap(cm,
            annot=True, fmt='d',
            cmap='RdYlGn',
            linewidths=0.5,
            linecolor='#0F1117',
            xticklabels=['No Churn', 'Churned'],
            yticklabels=['No Churn', 'Churned'],
            ax=ax)
ax.set_title(f'{best_name} — Confusion Matrix',
             fontsize=14, fontweight='bold',
             color=COLORS['text'], pad=15)
ax.set_xlabel('Predicted', fontsize=11, color=COLORS['subtext'])
ax.set_ylabel('Actual',    fontsize=11, color=COLORS['subtext'])
plt.tight_layout()
plt.savefig(data_path + 'plot7_confusion_matrix.png',
            dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.show()
print("Plot 7 saved!")

# ══════════════════════════════════════════════════════
# STEP 5 — ROC Curve
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(COLORS['bg'])

plot_colors = [COLORS['blue'], COLORS['purple'], COLORS['orange']]
for (name, res), color in zip(results.items(), plot_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC = {res['roc']:.3f})")

ax.plot([0,1],[0,1], color='#555566',
        linestyle='--', lw=1, label='Random Classifier')
ax.set_title('ROC Curve — All Models',
             fontsize=14, fontweight='bold',
             color=COLORS['text'], pad=15)
ax.set_xlabel('False Positive Rate', fontsize=11, color=COLORS['subtext'])
ax.set_ylabel('True Positive Rate',  fontsize=11, color=COLORS['subtext'])
ax.yaxis.grid(True, zorder=0)
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.legend(fontsize=10,
          facecolor=COLORS['card'],
          edgecolor='none',
          labelcolor=COLORS['text'])
plt.tight_layout()
plt.savefig(data_path + 'plot8_roc_curve.png',
            dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.show()
print("Plot 8 saved!")

# ══════════════════════════════════════════════════════
# STEP 6 — Feature Importance
# ══════════════════════════════════════════════════════
rf_model  = results['Random Forest']['model']
feat_imp  = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(COLORS['bg'])
bars = ax.barh(feat_imp.index, feat_imp.values,
               color=COLORS['purple'], zorder=3)
for bar, val in zip(bars, feat_imp.values):
    ax.text(val + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}',
            va='center', fontsize=9,
            color=COLORS['text'])
ax.set_title('Top 15 Feature Importances — Random Forest',
             fontsize=14, fontweight='bold',
             color=COLORS['text'], pad=15)
ax.set_xlabel('Importance Score', fontsize=11, color=COLORS['subtext'])
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.tight_layout()
plt.savefig(data_path + 'plot9_feature_importance.png',
            dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.show()
print("Plot 9 saved!")

# ══════════════════════════════════════════════════════
# STEP 7 — Save Best Model
# ══════════════════════════════════════════════════════
best_model = results[best_name]['model']
joblib.dump(best_model, model_path + 'best_model.pkl')
joblib.dump(sm,         model_path + 'smote.pkl')
print(f"\nBest model saved: {best_name}")

# ══════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════
print("\n" + "="*50)
print("="*50)
for name, res in results.items():
    print(f"\n  {name}")
    print(f"  Accuracy : {res['acc']*100:.2f}%")
    print(f"  ROC-AUC  : {res['roc']:.4f}")
print(f"\n  Classification Report — {best_name}:")
print(classification_report(y_test,
      results[best_name]['y_pred'],
      target_names=['No Churn', 'Churned']))
print("="*50)
print(f"  Best model saved → models/best_model.pkl")
