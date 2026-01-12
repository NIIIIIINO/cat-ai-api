import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

IDENTITY_THRESHOLD = 0.78

def identify_cat(query_emb, cat_bank):
    best_score = 0.0
    best_cat = None

    for cat_uid, emb_list in cat_bank.items():
        if not emb_list:
            continue

        sims = cosine_similarity([query_emb], emb_list)[0]
        score = float(np.max(sims))

        if score > best_score:
            best_score = score
            best_cat = cat_uid

    if best_score >= IDENTITY_THRESHOLD:
        return best_cat, best_score

    return None, best_score
