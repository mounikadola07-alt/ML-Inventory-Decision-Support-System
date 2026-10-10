import json

with open("experiment_results.json") as f:
    res = json.load(f)

for ds_name, ds_data in res.items():
    print("=" * 80)
    print("DATASET:", ds_data["Dataset"])
    print("TARGET:", ds_data["Target"])
    print("HORIZON:", ds_data["Evaluation_Horizon"])
    print("-" * 80)
    print(f"{'Model':22s} | {'MAE':10s} | {'RMSE':10s} | {'MAPE (%)':10s} | {'R2':8s} | {'Bias (%)':10s}")
    print("-" * 80)
    for m in ds_data["Holdout_Metrics"]:
        mape_val = m.get("MAPE_Filtered")
        mape_str = f"{mape_val:.2f}" if mape_val is not None else "N/A"
        print(f"{m['Model']:22s} | {m['MAE']:10.2f} | {m['RMSE']:10.2f} | {mape_str:10s} | {m['R2']:8.4f} | {m['Bias_Percentage']:10.2f}")
    
    # Classification metrics
    clf_key = None
    for k in ["Classification_Surge_Metrics", "Classification_Occurrence_Metrics"]:
        if k in ds_data:
            clf_key = k
            break
    if clf_key:
        print("\n--- CLASSIFICATION & STOCKOUT RISK METRICS ---")
        print(f"{'Classifier':22s} | {'ROC AUC':8s} | {'PR AUC':8s} | {'Accuracy (%)':12s} | {'F1':6s} | {'Precision':10s} | {'Recall':8s}")
        print("-" * 85)
        for c in ds_data[clf_key]:
            print(f"{c['Model']:22s} | {c['ROC_AUC']:8.4f} | {c['PR_AUC']:8.4f} | {c['Accuracy']*100:11.2f}% | {c['F1_Score']:6.4f} | {c['Precision']:10.4f} | {c['Recall']:8.4f}")

    if "Inventory_Comparison_Store1" in ds_data:
        print("\n--- INVENTORY COMPARISON (Store 1) ---")
        inv = ds_data["Inventory_Comparison_Store1"]
        for pol, d in inv.items():
            print(f"  {pol:22s}: SS={d.get('Safety_Stock')}, ROP={d.get('ROP')}, Total Cost={d.get('Total_Cost')}, Fill Rate={d.get('Fill_Rate')}")
