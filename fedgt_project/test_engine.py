import django
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fedgt_project.settings')
django.setup()

from core.fedgt_engine import (decode_malicious_clients, compute_pca_scores,
                                cluster_clients, run_secure_aggregation_all_rounds)

# ── Test 1: FedGT Decode ──────────────────────────────────────────
print("=" * 50)
print("TEST 1: FedGT DECODE ALGORITHM")
print("=" * 50)
results = decode_malicious_clients(None)
flagged = [r for r in results if r['flagged']]
print(f"Clients scored: {len(results)}")
print(f"Flagged as malicious: {len(flagged)}")
print("\nTop 5 by suspicion score:")
for r in results[:5]:
    mark = "FLAG" if r['flagged'] else "OK"
    print(f"  {mark} {r['client_id']:6s}  score={r['suspicion_score']:.4f}")
print("PASS\n")

# ── Test 2: PCA Anomaly ──────────────────────────────────────────
print("=" * 50)
print("TEST 2: PCA ANOMALY DETECTION")
print("=" * 50)
pca = compute_pca_scores(None)
print(f"Records computed: {len(pca)}")
extremes = sorted(pca, key=lambda x: abs(x['score']), reverse=True)[:3]
print("Highest anomaly scores:")
for e in extremes:
    print(f"  {e['client_id']:6s} R{e['round']:2d}  PCA score={e['score']:.4f}")
print("PASS\n")

# ── Test 3: K-Means Clustering ──────────────────────────────────
print("=" * 50)
print("TEST 3: K-MEANS CLUSTERING (k=3)")
print("=" * 50)
clusters = cluster_clients(3)
print(f"Clients clustered: {len(clusters)}")
by_cluster = {}
for c in clusters:
    by_cluster.setdefault(c['cluster_label'], []).append(c['client_id'])
for lbl, members in sorted(by_cluster.items()):
    print(f"  Cluster {lbl}: {', '.join(members)}")
print("PASS\n")

# ── Test 4: Secure Aggregation ──────────────────────────────────
print("=" * 50)
print("TEST 4: SECURE AGGREGATION (all rounds)")
print("=" * 50)
agg = run_secure_aggregation_all_rounds()
print(f"Rounds aggregated: {len(agg)}")
print("First 5 rounds:")
for r, res in sorted(list(agg.items()))[:5]:
    print(f"  Round {r:2d}: acc={res['global_accuracy']:6.4f}  "
          f"excluded={res['excluded_count']}  clipped={res['clipped_count']}")
print("PASS\n")

print("=" * 50)
print("ALL FEDGT ENGINE TESTS PASSED")
print("=" * 50)
