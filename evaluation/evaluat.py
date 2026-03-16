import pandas as pd
import json
import re

# 1. Load ground truth labels
with open("../data/3dmodel_query_matches.json", 'r', encoding='utf-8') as f:
    gt_data = json.load(f)
Gt_matches = [item["matches"] for item in gt_data]

count_zero = 0
count_one = 0
count_more = 0
for sublist in Gt_matches:
    length = len(sublist)
    if length == 0:
        count_zero += 1
    elif length == 1:
        count_one += 1
    else:
        count_more += 1
print("Total number of queries:", len(Gt_matches))
print(f"Number of queries with 0 matches: {count_zero}")
print(f"Number of queries with 1 match: {count_one}")
print(f"Number of queries with more than 1 match: {count_more}")

# 2. Load predicted labels
with open(r'../main/Pre_matches.json', 'r') as f:
    Pre_matches = json.load(f)
match_0_idxes = [idx for idx, sublist in enumerate(Gt_matches) if len(sublist) == 0]
Gt_matches_0 = [Gt_matches[i] for i in match_0_idxes]
Pre_matches_0 = [Pre_matches[i] for i in match_0_idxes]
error_retrieval = [Pre_matches[i] for i in match_0_idxes if len(Pre_matches[i]) != 0]
FPR = len(error_retrieval) / len(match_0_idxes)
Mean_error_num = (sum(len(sublist) for sublist in error_retrieval)) / len(error_retrieval)
print(f"False Positive Rate: {FPR}")
print(f"Average number of false matches: {Mean_error_num}")


match_no_0_idxes = [idx for idx, sublist in enumerate(Gt_matches) if len(sublist) != 0]
Gt_matches_no_0 = [Gt_matches[i] for i in match_no_0_idxes]
Pre_matches_no_0 = [Pre_matches[i] for i in match_no_0_idxes]

# 3. Evaluate retrieval accuracy
All_pre = []
All_rec = []
for i in range(len(Pre_matches_no_0)):
    TP = 0
    for element in Pre_matches_no_0[i]:
        if element in Gt_matches_no_0[i]:
            TP += 1
    pre = TP / max(0.1, len(Pre_matches_no_0[i]))
    rec = TP / len(Gt_matches_no_0[i])
    All_pre.append(pre)
    All_rec.append(rec)

Avg_pre = sum(All_pre) / len(All_pre)
Avg_rec = sum(All_rec) / len(All_rec)
Avg_F1 = 2 * Avg_pre * Avg_rec / (Avg_pre + Avg_rec)

print('Average Precision:', Avg_pre)
print('Average Recall:', Avg_rec)
print('Average F1 Score:', Avg_F1)