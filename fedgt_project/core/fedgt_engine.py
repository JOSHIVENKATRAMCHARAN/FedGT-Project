"""
fedgt_engine.py
===============
Real algorithmic implementations for the FedGT federated learning security system.

Pillars:
  1. FedGT Decoding   — infer malicious clients from binary group test results
  2. PCA Anomaly      — compute live PCA-based anomaly scores from model updates
  3. K-Means Cluster  — cluster clients by update behaviour
  4. Secure Aggregation — FedAvg with norm-clipping + exclusion of flagged clients
"""

import numpy as np
import json
from collections import defaultdict
from django.db import transaction


# ─── 1. FedGT DECODING ALGORITHM ──────────────────────────────────────────────

def decode_malicious_clients(round_number=None):
    """
    FedGT group-testing decoding.

    For each client, compute a suspicion score = sum over all groups the client
    belongs to of (test_result / group_size).  A client that consistently
    appears in *positive* (suspicious) groups gets a high score.

    Groups with test_result=1 are considered suspicious (low utility / anomalous
    gradient behaviour flagged by the server's group test).

    Args:
        round_number: int or None. If None, aggregates across ALL rounds.

    Returns:
        list of dicts: [{client_id, suspicion_score, flagged}, ...]
        sorted by descending suspicion_score.
    """
    from .models import GroupAssignment, TestResult, FedGTDecodeResult

    # Build group membership map
    assignments = GroupAssignment.objects.all()
    group_members = defaultdict(list)       # group_id -> [client_id, ...]
    client_groups  = defaultdict(list)      # client_id -> [group_id, ...]
    for a in assignments:
        group_members[a.group_id].append(a.client_id)
        client_groups[a.client_id].append(a.group_id)

    all_client_ids = list(client_groups.keys())

    # Fetch test results
    qs = TestResult.objects.all()
    if round_number:
        qs = qs.filter(round=round_number)

    # Suspicion scoring
    # score(c) = Σ_{g ∋ c, test(g)=1} (1 / |g|)   — weighted by group size
    suspicion = defaultdict(float)
    participation = defaultdict(int)   # how many groups the client is in

    for tr in qs:
        members = group_members.get(tr.group_id, [])
        if not members:
            continue
        weight = 1.0 / len(members)
        for cid in members:
            participation[cid] += 1
            if tr.test_result == 1:
                suspicion[cid] += weight

    if not suspicion and participation:
        # All tests were 0 — still build zero-score entries
        for cid in all_client_ids:
            suspicion[cid] = 0.0

    # Normalise: divide by total rounds of participation to get average suspicion
    scores = {}
    for cid in all_client_ids:
        p = participation.get(cid, 1)
        scores[cid] = round(suspicion.get(cid, 0.0) / p, 4)

    if not scores:
        return []

    # Threshold: flag clients above mean + 0.5*std
    vals = list(scores.values())
    threshold = np.mean(vals) + 0.5 * np.std(vals)

    results = []
    with transaction.atomic():
        for cid, score in sorted(scores.items(), key=lambda x: -x[1]):
            flagged = score >= threshold
            FedGTDecodeResult.objects.update_or_create(
                client_id=cid,
                round_number=round_number or 0,
                defaults={'suspicion_score': score, 'flagged': flagged}
            )
            results.append({'client_id': cid, 'suspicion_score': score, 'flagged': flagged})

    return results


# ─── 2. PCA ANOMALY DETECTION ─────────────────────────────────────────────────

def compute_pca_scores(round_number=None):
    """
    Compute PCA-based anomaly scores from ModelUpdate feature vectors.

    Feature vector per update: [loss, accuracy, update_norm, gradient_variance]
    Project onto 1st principal component → large absolute deviation = anomaly.

    Args:
        round_number: int or None. If None, uses ALL rounds.

    Returns:
        list of dicts: [{client_id, round, score}, ...]
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from .models import ModelUpdate, PCAScore

    qs = ModelUpdate.objects.all()
    if round_number:
        qs = qs.filter(round=round_number)

    records = list(qs.values('round', 'client_id', 'loss', 'accuracy',
                              'update_norm', 'gradient_variance'))
    if len(records) < 2:
        return []

    X = np.array([[r['loss'], r['accuracy'], r['update_norm'],
                   r['gradient_variance']] for r in records])

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA to 1D
    pca = PCA(n_components=1)
    scores_1d = pca.fit_transform(X_scaled).flatten()   # shape (N,)

    results = []
    with transaction.atomic():
        for i, rec in enumerate(records):
            score = round(float(scores_1d[i]), 4)
            PCAScore.objects.update_or_create(
                client_id=rec['client_id'],
                round=rec['round'],
                defaults={'score': score}
            )
            results.append({'client_id': rec['client_id'],
                            'round': rec['round'],
                            'score': score})

    return results


# ─── 3. K-MEANS CLUSTERING ────────────────────────────────────────────────────

def cluster_clients(n_clusters=3):
    """
    Cluster clients by their average update behaviour across all rounds.

    Feature vector per client: [mean_loss, mean_accuracy, mean_update_norm,
                                 mean_gradient_variance]
    Runs K-Means, then projects to 2D via PCA for visualisation.

    Args:
        n_clusters: number of K-Means clusters (default 3).

    Returns:
        list of dicts: [{client_id, cluster_label, pca_x, pca_y}, ...]
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from .models import ModelUpdate, ClientCluster

    # Aggregate per-client stats
    client_stats = defaultdict(lambda: {'loss': [], 'accuracy': [],
                                         'update_norm': [], 'gradient_variance': []})
    for u in ModelUpdate.objects.all():
        s = client_stats[u.client_id]
        s['loss'].append(u.loss)
        s['accuracy'].append(u.accuracy)
        s['update_norm'].append(u.update_norm)
        s['gradient_variance'].append(u.gradient_variance)

    if not client_stats:
        return []

    client_ids = sorted(client_stats.keys())
    X = np.array([
        [np.mean(client_stats[c]['loss']),
         np.mean(client_stats[c]['accuracy']),
         np.mean(client_stats[c]['update_norm']),
         np.mean(client_stats[c]['gradient_variance'])]
        for c in client_ids
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    k = min(n_clusters, len(client_ids))
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X_scaled)

    # 2D PCA for scatter
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    results = []
    with transaction.atomic():
        for i, cid in enumerate(client_ids):
            ClientCluster.objects.update_or_create(
                client_id=cid,
                defaults={
                    'cluster_label': int(labels[i]),
                    'pca_x': round(float(coords[i, 0]), 4),
                    'pca_y': round(float(coords[i, 1]), 4),
                }
            )
            results.append({
                'client_id': cid,
                'cluster_label': int(labels[i]),
                'pca_x': round(float(coords[i, 0]), 4),
                'pca_y': round(float(coords[i, 1]), 4),
            })

    return results


# ─── 4. SECURE AGGREGATION (FedAvg + norm-clipping) ──────────────────────────

def run_secure_aggregation(round_number):
    """
    Simulate secure FedAvg aggregation for a given round:
      1. Retrieve all ModelUpdate records for the round.
      2. Identify flagged clients from FedGTDecodeResult.
      3. Clip updates whose update_norm exceeds the clipping threshold
         (median + 2 × IQR of norms among included clients).
      4. Compute weighted-average global accuracy and loss (weight = 1/update_norm,
         capped at a minimum to avoid division by zero).
      5. Store result in AggregationResult.

    Args:
        round_number: int

    Returns:
        dict with keys: global_accuracy, global_loss, excluded_count, clipped_count,
                        included_count, excluded_clients
    """
    from .models import ModelUpdate, FedGTDecodeResult, AggregationResult

    updates = list(ModelUpdate.objects.filter(round=round_number))
    if not updates:
        return None

    # Flagged clients (aggregated decode across all rounds stored)
    flagged_ids = set(
        FedGTDecodeResult.objects.filter(flagged=True)
                                  .values_list('client_id', flat=True)
    )

    excluded = [u for u in updates if u.client_id in flagged_ids]
    included = [u for u in updates if u.client_id not in flagged_ids]

    if not included:
        included = updates   # safety fallback

    norms = np.array([u.update_norm for u in included])

    # Clipping threshold: median + 2*IQR
    q75, q25 = np.percentile(norms, [75, 25])
    iqr = q75 - q25
    clip_threshold = float(np.median(norms) + 2 * iqr)

    clipped_count = 0
    clipped_updates = []
    for u in included:
        if u.update_norm > clip_threshold:
            # Scale down accuracy/loss impact proportionally
            scale = clip_threshold / u.update_norm
            clipped_updates.append({
                'accuracy': u.accuracy * scale,
                'loss': u.loss * scale,
                'weight': 1.0
            })
            clipped_count += 1
        else:
            clipped_updates.append({
                'accuracy': u.accuracy,
                'loss': u.loss,
                'weight': 1.0
            })

    total_weight = sum(c['weight'] for c in clipped_updates)
    global_accuracy = round(
        sum(c['accuracy'] * c['weight'] for c in clipped_updates) / total_weight, 4
    )
    global_loss = round(
        sum(c['loss'] * c['weight'] for c in clipped_updates) / total_weight, 6
    )

    excluded_client_ids = [u.client_id for u in excluded]

    AggregationResult.objects.update_or_create(
        round=round_number,
        defaults={
            'global_accuracy': global_accuracy,
            'global_loss': global_loss,
            'excluded_clients_json': json.dumps(excluded_client_ids),
            'clipped_count': clipped_count,
            'included_count': len(included),
        }
    )

    return {
        'global_accuracy': global_accuracy,
        'global_loss': global_loss,
        'excluded_count': len(excluded),
        'clipped_count': clipped_count,
        'included_count': len(included),
        'excluded_clients': excluded_client_ids,
    }


def run_secure_aggregation_all_rounds():
    """Run secure aggregation for every available round."""
    from .models import ModelUpdate
    rounds = sorted(ModelUpdate.objects.values_list('round', flat=True).distinct())
    results = {}
    for r in rounds:
        res = run_secure_aggregation(r)
        if res:
            results[r] = res
    return results
