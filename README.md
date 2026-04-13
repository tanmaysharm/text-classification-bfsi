# BFSI Multi-Label Text Classification

NLP pipeline for classifying financial investigation alerts into 20 operational categories using TF-IDF + XGBoost + OneVsRest.

## Problem
A single alert message (e.g., "unusual wire transfer flagged for AML review") can trigger multiple response workflows simultaneously. Multi-label classification handles this where single-label models fail.

## Approach
- **Features**: TF-IDF (unigrams + bigrams, 15k features, sublinear TF)
- **Model**: XGBClassifier wrapped in OneVsRestClassifier
- **Imbalance handling**: scale_pos_weight per label
- **Evaluation**: Micro-F1, Macro-F1, Hamming Loss

## Results
| Metric | Score |
|--------|-------|
| Micro-F1 | 0.990 |
| Macro-F1 | ~0.99 |
| Hamming Loss | ~0.001 |

> Note: High scores reflect synthetic templated data. Production scores on real BFSI text expected at 0.75–0.85 Micro-F1.

## Categories (20)
wire_transfer, card_dispute, account_takeover, velocity_alert, location_mismatch, unusual_amount, new_payee_risk, dormant_account, kyc_flag, aml_alert, data_quality_issue, system_error, high_priority, manual_review_required, escalation_needed, corporate_client, retail_client, compliance_breach, false_positive, retraining_trigger

## Stack
Python · scikit-learn · XGBoost · TF-IDF · OneVsRestClassifier · matplotlib
