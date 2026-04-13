
# ============================================================
# NLP Multi-Label Text Classification — BFSI Use Case
# Stack: TF-IDF + XGBoost + OneVsRestClassifier
# Evaluation: micro-F1, Hamming Loss
# Author: Tanmay Sharma
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, hamming_loss,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

print("=" * 60)
print("BFSI Multi-Label Text Classification")
print("=" * 60)

# ============================================================
# SECTION 1 — SYNTHETIC BFSI DATASET
# 20 categories, 1200 samples, realistic operational text
# ============================================================

CATEGORIES = [
    'wire_transfer', 'card_dispute', 'account_takeover',
    'velocity_alert', 'location_mismatch', 'unusual_amount',
    'new_payee_risk', 'dormant_account', 'kyc_flag',
    'aml_alert', 'data_quality_issue', 'system_error',
    'high_priority', 'manual_review_required', 'escalation_needed',
    'corporate_client', 'retail_client', 'compliance_breach',
    'false_positive', 'retraining_trigger'
]

# Seed text patterns per category
TEMPLATES = {
    'wire_transfer': [
        "International wire transfer of {amt} to {country} account flagged for review",
        "Wire transfer initiated to new beneficiary in {country} amount {amt}",
        "Large outgoing wire {amt} to first-time recipient requires verification",
    ],
    'card_dispute': [
        "Customer disputes transaction {amt} at {merchant} on {date}",
        "Chargeback requested for card payment {amt} merchant category mismatch",
        "Card transaction disputed customer claims unauthorised {amt} charge",
    ],
    'account_takeover': [
        "Email and phone changed followed by large transfer within 2 hours",
        "Login from new device after multiple failed attempts account change detected",
        "Profile update on dormant account followed by immediate withdrawal attempt",
    ],
    'velocity_alert': [
        "{n} transactions within {hrs} hours breaching velocity threshold",
        "Card used {n} times in {hrs} hour window at different locations",
        "High frequency low value transactions detected possible structuring {n} txns",
    ],
    'location_mismatch': [
        "Card used in Mumbai and London within {hrs} hours impossible travel",
        "Transaction from IP address in {country} while customer is domestic",
        "Geographic anomaly card swiped {city1} then {city2} within {hrs} minutes",
    ],
    'unusual_amount': [
        "Transaction amount {amt} significantly above customer average of {avg}",
        "Large cash withdrawal {amt} exceeds monthly average by {pct} percent",
        "Unusual credit amount {amt} received from unknown source requires review",
    ],
    'new_payee_risk': [
        "First payment to new beneficiary account {amt} high risk flag",
        "New payee added and immediate large transfer {amt} initiated",
        "Unverified payee receiving {amt} first time transfer risk assessment needed",
    ],
    'dormant_account': [
        "Account inactive for {months} months now showing large transaction",
        "Dormant account reactivated with {amt} deposit followed by immediate withdrawal",
        "No activity for {months} months sudden {amt} transfer triggers review",
    ],
    'kyc_flag': [
        "Customer KYC documentation expired requires update before processing",
        "Enhanced due diligence required PEP status detected for account",
        "KYC refresh overdue high risk customer category requires immediate action",
    ],
    'aml_alert': [
        "Transaction pattern consistent with layering multiple accounts {amt}",
        "Cash structuring detected multiple deposits just below reporting threshold",
        "Suspicious activity report required unusual cash movement {amt} in {days} days",
    ],
    'data_quality_issue': [
        "Missing customer identifier in transaction record batch {batch}",
        "Duplicate transaction ID detected in payment file requires investigation",
        "Data field mismatch account number format invalid in submission",
    ],
    'system_error': [
        "Payment processing failed due to timeout error transaction {txn} incomplete",
        "Core banking system error account balance mismatch detected",
        "API integration failure downstream system not responding batch {batch}",
    ],
    'high_priority': [
        "Critical alert requires immediate analyst attention SLA breach imminent",
        "Priority 1 case escalated senior review needed within {hrs} hours",
        "Urgent flag regulatory deadline approaching case {case} unresolved",
    ],
    'manual_review_required': [
        "Automated decision confidence below threshold human review needed",
        "Edge case detected model confidence {pct} percent analyst assignment required",
        "Complex scenario outside rule parameters manual assessment necessary",
    ],
    'escalation_needed': [
        "Case escalated to L2 analyst initial review inconclusive",
        "Customer complaint unresolved requires supervisor intervention",
        "Regulatory breach confirmed immediate escalation to compliance team",
    ],
    'corporate_client': [
        "Corporate account bulk payment {amt} multiple beneficiaries requires approval",
        "Business client treasury operation large volume transaction {amt}",
        "Corporate sweep account movement {amt} end of day reconciliation",
    ],
    'retail_client': [
        "Retail customer personal account transaction {amt} flagged",
        "Individual account holder {amt} savings withdrawal unusual pattern",
        "Consumer banking alert personal loan repayment missed {amt}",
    ],
    'compliance_breach': [
        "Regulatory limit exceeded transaction {amt} above permissible threshold",
        "FEMA violation suspected cross border transfer documentation missing",
        "RBI reporting threshold breached {amt} mandatory disclosure required",
    ],
    'false_positive': [
        "Alert reviewed and confirmed legitimate transaction no action required",
        "Customer verified transaction {amt} whitelist updated false alert",
        "Rule triggered but business justification accepted closing case",
    ],
    'retraining_trigger': [
        "Model accuracy drop detected feature drift PSI score above threshold",
        "False positive rate increased {pct} percent last {days} days retraining needed",
        "Concept drift confirmed in {feature} distribution model refresh required",
    ],
}

def fill_template(template):
    """Fill placeholder values in templates"""
    return (template
        .replace('{amt}',     np.random.choice(['₹45,000', '$12,500', '€8,000', '₹2,10,000', '$75,000']))
        .replace('{country}', np.random.choice(['UAE', 'USA', 'Singapore', 'UK', 'China']))
        .replace('{merchant}',np.random.choice(['Amazon', 'unknown vendor', 'overseas retailer', 'crypto exchange']))
        .replace('{date}',    np.random.choice(['12-Apr-2026', '08-Mar-2026', '01-Apr-2026']))
        .replace('{n}',       str(np.random.randint(5, 20)))
        .replace('{hrs}',     str(np.random.randint(1, 6)))
        .replace('{city1}',   np.random.choice(['Mumbai', 'Delhi', 'Bangalore']))
        .replace('{city2}',   np.random.choice(['London', 'Dubai', 'New York']))
        .replace('{avg}',     np.random.choice(['₹5,000', '$2,000', '₹15,000']))
        .replace('{pct}',     str(np.random.randint(150, 400)))
        .replace('{months}',  str(np.random.randint(6, 24)))
        .replace('{days}',    str(np.random.randint(7, 30)))
        .replace('{batch}',   f'BATCH-{np.random.randint(1000,9999)}')
        .replace('{txn}',     f'TXN-{np.random.randint(100000,999999)}')
        .replace('{case}',    f'CASE-{np.random.randint(1000,9999)}')
        .replace('{feature}', np.random.choice(['transaction_amount', 'merchant_category', 'location']))
    )

# ── Generate dataset ──────────────────────────────────────────
np.random.seed(42)
records = []

for _ in range(1200):
    # Each sample gets 1–3 random categories (multi-label)
    n_labels = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
    chosen_cats = np.random.choice(CATEGORIES, size=n_labels, replace=False).tolist()

    # Build text from one primary category template
    primary = chosen_cats[0]
    template = np.random.choice(TEMPLATES[primary])
    text = fill_template(template)

    # Add context from secondary categories
    for extra_cat in chosen_cats[1:]:
        extra_tmpl = np.random.choice(TEMPLATES[extra_cat])
        text += '. ' + fill_template(extra_tmpl)

    records.append({'text': text, 'labels': chosen_cats})

df = pd.DataFrame(records)
print(f"\n✅ Dataset generated: {len(df)} samples, {len(CATEGORIES)} categories")
print(f"   Multi-label distribution: {df['labels'].apply(len).value_counts().to_dict()}")

# ============================================================
# SECTION 2 — PREPROCESSING + ENCODING
# ============================================================
mlb = MultiLabelBinarizer(classes=CATEGORIES)
Y = mlb.fit_transform(df['labels'])
X = df['text']

print(f"\n✅ Label matrix shape: {Y.shape}")
print(f"   Categories: {len(mlb.classes_)}")

# Class distribution
label_counts = pd.Series(Y.sum(axis=0), index=mlb.classes_).sort_values(ascending=False)
print(f"\n   Top 5 categories by frequency:")
print(label_counts.head())
print(f"   Min samples: {label_counts.min()} | Max: {label_counts.max()}")

# Train/test split — stratification not possible for multi-label, use random
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# SECTION 3 — TF-IDF VECTORISATION
# ============================================================
print("\n── TF-IDF Vectorisation ─────────────────────────────")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=15000,       # top 15k features
    sublinear_tf=True,        # log scaling
    min_df=2,                 # ignore very rare terms
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{2,}',  # min 2 chars per token
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"✅ TF-IDF matrix: {X_train_tfidf.shape}")
print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")

# ============================================================
# SECTION 4 — MODEL: XGBoost + OneVsRestClassifier
# ============================================================
print("\n── Model Training: XGBoost + OneVsRest ──────────────")

# Compute class weights to handle imbalance
label_freq = Y_train.sum(axis=0)
total = Y_train.shape[0]
scale_pos_weight = (total - label_freq) / (label_freq + 1e-6)
mean_spw = float(np.mean(scale_pos_weight))

xgb_base = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=mean_spw,    # handles class imbalance
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

model = OneVsRestClassifier(xgb_base, n_jobs=-1)
model.fit(X_train_tfidf, Y_train)
print("✅ Model trained")

# ============================================================
# SECTION 5 — EVALUATION
# ============================================================
print("\n── Evaluation ───────────────────────────────────────")

Y_pred = model.predict(X_test_tfidf)

micro_f1  = f1_score(Y_test, Y_pred, average='micro',   zero_division=0)
macro_f1  = f1_score(Y_test, Y_pred, average='macro',   zero_division=0)
h_loss    = hamming_loss(Y_test, Y_pred)

print(f"\n  Micro F1  : {micro_f1:.4f}   ← primary metric")
print(f"  Macro F1  : {macro_f1:.4f}")
print(f"  Hamming Loss: {h_loss:.4f}   ← lower is better")

# Per-category performance
per_cat_f1 = f1_score(Y_test, Y_pred, average=None, zero_division=0)
perf_df = pd.DataFrame({
    'Category': mlb.classes_,
    'F1_Score': per_cat_f1,
    'Support':  Y_test.sum(axis=0)
}).sort_values('F1_Score', ascending=False)

print(f"\n  Per-Category F1 (Top 10):")
print(perf_df.head(10).to_string(index=False))

# ============================================================
# SECTION 6 — VISUALISATIONS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('BFSI Multi-Label Text Classification Results', fontsize=14, fontweight='bold')

# Plot 1 — Category F1 scores
axes[0].barh(perf_df['Category'], perf_df['F1_Score'], color='steelblue')
axes[0].axvline(x=micro_f1, color='red', linestyle='--', label=f'Micro F1: {micro_f1:.3f}')
axes[0].set_xlabel('F1 Score')
axes[0].set_title('Per-Category F1 Score')
axes[0].legend()
axes[0].set_xlim(0, 1)

# Plot 2 — Label frequency distribution
axes[1].bar(range(len(CATEGORIES)), label_counts.values, color='darkorange')
axes[1].set_xticks(range(len(CATEGORIES)))
axes[1].set_xticklabels(label_counts.index, rotation=90, fontsize=8)
axes[1].set_ylabel('Sample Count')
axes[1].set_title('Label Distribution (Class Imbalance View)')

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved: classification_results.png")

# ============================================================
# SECTION 7 — INFERENCE DEMO
# ============================================================
print("\n── Live Inference Demo ──────────────────────────────")

test_cases = [
    "Customer account shows 8 transactions in 2 hours from different locations",
    "Wire transfer of $45,000 to new international beneficiary in UAE",
    "Model accuracy dropped 15 percent in last 7 days feature drift detected",
    "KYC documents expired PEP status customer enhanced due diligence required",
]

for text in test_cases:
    vec  = tfidf.transform([text])
    pred = model.predict(vec)[0]
    cats = [mlb.classes_[i] for i, v in enumerate(pred) if v == 1]
    print(f"\n  Input : {text[:70]}...")
    print(f"  Labels: {cats if cats else ['none_detected']}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Dataset     : 1,200 samples | {len(CATEGORIES)} categories")
print(f"  Features    : TF-IDF bigrams | {X_train_tfidf.shape[1]:,} features")
print(f"  Model       : XGBoost + OneVsRestClassifier")
print(f"  Imbalance   : Handled via scale_pos_weight")
print(f"  Micro F1    : {micro_f1:.4f}")
print(f"  Hamming Loss: {h_loss:.4f}")
print("\n✅ Done. Push to GitHub.")