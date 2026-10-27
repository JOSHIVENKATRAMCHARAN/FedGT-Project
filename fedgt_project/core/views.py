import json
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .models import (Client, ClientData, PoisonedClient,
                     GroupAssignment, ModelUpdate, TestResult, TrainedModel,
                     FedGTDecodeResult, PCAScore, ClientCluster, AggregationResult)
from .data_loader import load_all_data
from .ml_engine import train_all_models, train_model, predict_sample, MODEL_LABELS
from .fedgt_engine import (decode_malicious_clients, compute_pca_scores,
                           cluster_clients, run_secure_aggregation,
                           run_secure_aggregation_all_rounds)
from training.pytorch_engine import run_local_training


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


# ─── AUTH ────────────────────────────────────────────────────────────────────

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        u = request.POST.get('username')
        p = request.POST.get('password')
        user = authenticate(request, username=u, password=p)
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid credentials.')
    return render(request, 'core/login.html')


@login_required
@require_POST
def logout_view(request):
    logout(request)
    return redirect('login')


# ─── DASHBOARD ───────────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    load_all_data()
    total_clients = Client.objects.count()
    malicious_clients = Client.objects.filter(is_malicious=True).count()
    benign_clients = total_clients - malicious_clients
    total_samples = ClientData.objects.count()
    total_groups = GroupAssignment.objects.values('group_id').distinct().count()
    trained_models = TrainedModel.objects.all()
    best_model = TrainedModel.objects.order_by('-accuracy').first()
    latest_round = ModelUpdate.objects.order_by('-round').values_list('round', flat=True).first() or 0

    # Device type pie
    device_counts = {}
    for c in Client.objects.values('device_type'):
        device_counts[c['device_type']] = device_counts.get(c['device_type'], 0) + 1
    device_labels = list(device_counts.keys())
    device_values = list(device_counts.values())

    # Region bar
    region_counts = {}
    for c in Client.objects.values('region'):
        r = c['region']
        region_counts[r] = region_counts.get(r, 0) + 1

    ctx = {
        'total_clients': total_clients,
        'malicious_clients': malicious_clients,
        'benign_clients': benign_clients,
        'total_samples': total_samples,
        'total_groups': total_groups,
        'trained_models': trained_models,
        'best_model': best_model,
        'latest_round': latest_round,
        'device_labels': json.dumps(device_labels),
        'device_values': json.dumps(device_values),
        'region_labels': json.dumps(list(region_counts.keys())),
        'region_values': json.dumps(list(region_counts.values())),
        'malicious_pie': json.dumps([benign_clients, malicious_clients]),
    }
    return render(request, 'core/dashboard.html', ctx)


# ─── CLIENT MONITOR ──────────────────────────────────────────────────────────

@login_required
def client_monitor(request):
    clients = Client.objects.all()
    poisoned = {p.client_id: p for p in PoisonedClient.objects.all()}
    client_list = []
    for c in clients:
        latest = ModelUpdate.objects.filter(client_id=c.client_id).order_by('-round').first()
        client_list.append({
            'client': c,
            'attack': poisoned.get(c.client_id),
            'latest_update': latest,
        })

    # K-Means cluster data for scatter plot
    clusters = ClientCluster.objects.all()
    cluster_colors = ['#4f46e5', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4']
    cluster_series = {}
    for cl in clusters:
        lbl = cl.cluster_label
        if lbl not in cluster_series:
            cluster_series[lbl] = {'label': f'Cluster {lbl}', 'data': [],
                                   'backgroundColor': cluster_colors[lbl % len(cluster_colors)]}
        cluster_series[lbl]['data'].append({'x': cl.pca_x, 'y': cl.pca_y, 'label': cl.client_id})
    cluster_datasets = list(cluster_series.values())

    return render(request, 'core/client_monitor.html', {
        'client_list': client_list,
        'clusters': clusters,
        'cluster_datasets_json': json.dumps(cluster_datasets),
    })


# ─── GROUP VIEWER ────────────────────────────────────────────────────────────

@login_required
def group_viewer(request):
    assignments = GroupAssignment.objects.all()
    groups = defaultdict(list)
    for a in assignments:
        groups[a.group_id].append(a.client_id)

    malicious_ids = set(Client.objects.filter(is_malicious=True).values_list('client_id', flat=True))
    test_results = TestResult.objects.all().order_by('round', 'group_id')

    # Group utility line chart data per round
    rounds = sorted(TestResult.objects.values_list('round', flat=True).distinct())
    group_ids = sorted(groups.keys())

    utility_by_group = {}
    for g in group_ids:
        utility_by_group[g] = []
        for r in rounds:
            tr = TestResult.objects.filter(round=r, group_id=g).first()
            utility_by_group[g].append(tr.utility_score if tr else 0)

    group_data = []
    for g, clients in sorted(groups.items()):
        mal = [c for c in clients if c in malicious_ids]
        latest_tr = TestResult.objects.filter(group_id=g).order_by('-round').first()
        group_data.append({
            'group_id': g,
            'clients': clients,
            'malicious': mal,
            'is_suspicious': len(mal) > 0,
            'latest_result': latest_tr,
        })

    ctx = {
        'group_data': group_data,
        'test_results': test_results[:40],
        'rounds': json.dumps(rounds),
        'group_ids': json.dumps(group_ids),
        'utility_by_group': json.dumps(utility_by_group),
    }
    return render(request, 'core/group_viewer.html', ctx)


# ─── TRAINING DASHBOARD ──────────────────────────────────────────────────────

@login_required
def training_dashboard(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'pytorch_train':
            round_raw = request.POST.get('round_number', '').strip()
            epochs_raw = request.POST.get('epochs', '2').strip()
            round_number = int(round_raw) if round_raw.isdigit() else None
            epochs = int(epochs_raw) if epochs_raw.isdigit() else 2
            result = run_local_training(round_number=round_number, epochs=epochs)
            messages.success(
                request,
                f"PyTorch local training complete for round {result['round']} "
                f"across {result['client_count']} clients."
            )
            return redirect('training_dashboard')

    updates = ModelUpdate.objects.all().order_by('round', 'client_id')
    rounds = sorted(ModelUpdate.objects.values_list('round', flat=True).distinct())
    clients = sorted(Client.objects.values_list('client_id', flat=True))
    malicious_ids = set(Client.objects.filter(is_malicious=True).values_list('client_id', flat=True))

    # Per-round average accuracy (benign vs malicious)
    benign_acc, malicious_acc = [], []
    for r in rounds:
        us = ModelUpdate.objects.filter(round=r)
        b = [u.accuracy for u in us if u.client_id not in malicious_ids]
        m = [u.accuracy for u in us if u.client_id in malicious_ids]
        benign_acc.append(round(sum(b)/len(b), 2) if b else 0)
        malicious_acc.append(round(sum(m)/len(m), 2) if m else 0)

    # Loss per round average
    avg_loss = []
    for r in rounds:
        us = ModelUpdate.objects.filter(round=r)
        vals = [u.loss for u in us]
        avg_loss.append(round(sum(vals)/len(vals), 4) if vals else 0)

    agg_results = AggregationResult.objects.all().order_by('round')

    ctx = {
        'rounds': json.dumps(rounds),
        'benign_acc': json.dumps(benign_acc),
        'malicious_acc': json.dumps(malicious_acc),
        'avg_loss': json.dumps(avg_loss),
        'recent_updates': updates[:30],
        'total_rounds': len(rounds),
        'total_clients': len(clients),
        'agg_results': agg_results,
    }
    return render(request, 'core/training_dashboard.html', ctx)


# ─── ML MODELS ───────────────────────────────────────────────────────────────

@login_required
def ml_models(request):
    if request.method == 'POST':
        model_key = request.POST.get('model_key', 'all')
        if model_key == 'all':
            train_all_models()
            messages.success(request, 'All models trained successfully!')
        else:
            train_model(model_key)
            messages.success(request, f'{MODEL_LABELS.get(model_key, model_key)} trained successfully!')
        return redirect('ml_models')

    trained = TrainedModel.objects.all().order_by('-accuracy')
    model_labels_list = list(MODEL_LABELS.items())

    # Build comparison chart data
    names = [MODEL_LABELS.get(m.name, m.name) for m in trained]
    accs = [m.accuracy for m in trained]
    precs = [m.precision for m in trained]
    recs = [m.recall for m in trained]
    f1s = [m.f1_score for m in trained]

    # Confusion matrix images
    cm_images = {}
    for m in trained:
        try:
            cm = np.array(json.loads(m.confusion_matrix_json))
            if cm.size == 0:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            unique_labels = sorted(set(
                ClientData.objects.values_list('label', flat=True)
            ))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=unique_labels, yticklabels=unique_labels)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'Confusion Matrix — {MODEL_LABELS.get(m.name, m.name)}', fontsize=13)
            cm_images[m.name] = fig_to_base64(fig)
        except Exception:
            pass

    ctx = {
        'trained': trained,
        'model_labels': model_labels_list,
        'chart_names': json.dumps(names),
        'chart_accs': json.dumps(accs),
        'chart_precs': json.dumps(precs),
        'chart_recs': json.dumps(recs),
        'chart_f1s': json.dumps(f1s),
        'cm_images': cm_images,
    }
    return render(request, 'core/ml_models.html', ctx)


# ─── ATTACK ANALYZER ─────────────────────────────────────────────────────────

@login_required
def attack_analyzer(request):
    poisoned = PoisonedClient.objects.all()
    malicious_clients = Client.objects.filter(is_malicious=True)

    attack_type_counts = {}
    for p in poisoned:
        attack_type_counts[p.attack_type] = attack_type_counts.get(p.attack_type, 0) + 1

    # Attack strength distribution
    strengths = [p.attack_strength for p in poisoned]

    # Rounds where malicious clients had high update_norm
    mal_ids = list(malicious_clients.values_list('client_id', flat=True))
    mal_updates = ModelUpdate.objects.filter(client_id__in=mal_ids).order_by('round')
    ben_updates = ModelUpdate.objects.exclude(client_id__in=mal_ids).order_by('round')

    rounds = sorted(ModelUpdate.objects.values_list('round', flat=True).distinct())
    mal_norm_per_round = []
    ben_norm_per_round = []
    for r in rounds:
        m = [u.update_norm for u in mal_updates if u.round == r]
        b = [u.update_norm for u in ben_updates if u.round == r]
        mal_norm_per_round.append(round(sum(m)/len(m), 4) if m else 0)
        ben_norm_per_round.append(round(sum(b)/len(b), 4) if b else 0)

    ctx = {
        'poisoned': poisoned,
        'malicious_clients': malicious_clients,
        'attack_type_labels': json.dumps(list(attack_type_counts.keys())),
        'attack_type_values': json.dumps(list(attack_type_counts.values())),
        'rounds': json.dumps(rounds),
        'mal_norm': json.dumps(mal_norm_per_round),
        'ben_norm': json.dumps(ben_norm_per_round),
    }
    return render(request, 'core/attack_analyzer.html', ctx)


# ─── DETECTION INTERFACE ─────────────────────────────────────────────────────

@login_required
def detection(request):
    result = None
    model_used = None
    features_input = []
    error = None

    if request.method == 'POST':
        raw = request.POST.get('features', '')
        try:
            features = [float(x.strip()) for x in raw.split(',') if x.strip()]
            if len(features) != 10:
                error = 'Please enter exactly 10 comma-separated feature values.'
            else:
                pred, best = predict_sample(features)
                if pred is None:
                    error = 'No trained model found. Please train models first.'
                else:
                    result = pred
                    model_used = best
                    features_input = features
        except ValueError:
            error = 'Invalid input. Enter numeric values separated by commas.'

    # Sample values from dataset for convenience
    sample_row = ClientData.objects.first()
    sample_vals = ''
    if sample_row:
        sample_vals = ','.join([
            str(sample_row.feature_1), str(sample_row.feature_2),
            str(sample_row.feature_3), str(sample_row.feature_4),
            str(sample_row.feature_5), str(sample_row.feature_6),
            str(sample_row.feature_7), str(sample_row.feature_8),
            str(sample_row.feature_9), str(sample_row.feature_10),
        ])

    ctx = {
        'result': result,
        'model_used': model_used,
        'features_input': features_input,
        'error': error,
        'sample_vals': sample_vals,
    }
    return render(request, 'core/detection.html', ctx)


# ─── REPORTS ─────────────────────────────────────────────────────────────────

@login_required
def reports(request):
    clients = Client.objects.all()
    total = clients.count()
    malicious = clients.filter(is_malicious=True).count()
    benign = total - malicious

    updates = ModelUpdate.objects.all()
    avg_acc = round(sum(u.accuracy for u in updates) / updates.count(), 2) if updates.count() else 0
    avg_loss = round(sum(u.loss for u in updates) / updates.count(), 4) if updates.count() else 0

    test_results = TestResult.objects.all()
    positive_tests = test_results.filter(test_result=1).count()
    negative_tests = test_results.filter(test_result=0).count()

    trained = TrainedModel.objects.all().order_by('-accuracy')

    ctx = {
        'total_clients': total,
        'malicious_clients': malicious,
        'benign_clients': benign,
        'avg_acc': avg_acc,
        'avg_loss': avg_loss,
        'positive_tests': positive_tests,
        'negative_tests': negative_tests,
        'trained_models': trained,
        'mal_pct': round(malicious / total * 100, 1) if total else 0,
        'positive_pct': round(positive_tests / test_results.count() * 100, 1) if test_results.count() else 0,
    }
    return render(request, 'core/reports.html', ctx)


# ─── FEDGT ENGINE DASHBOARD ──────────────────────────────────────────────────

@login_required
def fedgt_dashboard(request):
    """
    Unified FedGT Engine control panel.
    POST actions: decode | pca | cluster | aggregate_all
    """
    load_all_data()

    action_result = None
    action_label  = None

    available_rounds = sorted(
        ModelUpdate.objects.values_list('round', flat=True).distinct()
    )

    if request.method == 'POST':
        action = request.POST.get('action', '')
        selected_round = request.POST.get('round_number', '')
        round_num = int(selected_round) if selected_round.isdigit() else None

        if action == 'decode':
            results = decode_malicious_clients(round_num)
            action_label = f'FedGT Decode — Round {"All" if not round_num else round_num}'
            action_result = {'type': 'decode', 'data': results}
            messages.success(request, f'FedGT Decode complete — {len(results)} clients scored.')

        elif action == 'pca':
            results = compute_pca_scores(round_num)
            action_label = f'PCA Recompute — Round {"All" if not round_num else round_num}'
            action_result = {'type': 'pca', 'data': results[:40]}
            messages.success(request, f'PCA scores computed for {len(results)} records.')

        elif action == 'cluster':
            n_clusters = int(request.POST.get('n_clusters', 3))
            results = cluster_clients(n_clusters=n_clusters)
            action_label = f'K-Means Clustering (k={n_clusters})'
            action_result = {'type': 'cluster', 'data': results}
            messages.success(request, f'Clustered {len(results)} clients into {n_clusters} groups.')

        elif action == 'aggregate_all':
            run_secure_aggregation_all_rounds()
            messages.success(request, 'Secure aggregation run for all rounds.')
            return redirect('fedgt_dashboard')

        elif action == 'aggregate_round' and round_num:
            run_secure_aggregation(round_num)
            messages.success(request, f'Secure aggregation complete for Round {round_num}.')
            return redirect('fedgt_dashboard')

    # ── Fetch stored results for display ──
    decode_results = FedGTDecodeResult.objects.all().order_by('-suspicion_score')
    pca_latest = (
        PCAScore.objects
        .filter(round=available_rounds[-1])
        .order_by('-score') if available_rounds else PCAScore.objects.none()
    )
    clusters = ClientCluster.objects.all()
    agg_results = AggregationResult.objects.all().order_by('round')

    # Chart data for aggregation
    agg_rounds   = [a.round for a in agg_results]
    agg_acc      = [a.global_accuracy for a in agg_results]
    agg_loss     = [a.global_loss for a in agg_results]
    agg_excluded = [json.loads(a.excluded_clients_json) for a in agg_results]
    agg_clipped  = [a.clipped_count for a in agg_results]

    # Chart data for cluster scatter
    cluster_colors = ['#4f46e5', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4']
    cluster_series = {}
    for c in clusters:
        lbl = c.cluster_label
        if lbl not in cluster_series:
            cluster_series[lbl] = {'label': f'Cluster {lbl}',
                                   'data': [],
                                   'backgroundColor': cluster_colors[lbl % len(cluster_colors)]}
        cluster_series[lbl]['data'].append({'x': c.pca_x, 'y': c.pca_y,
                                             'label': c.client_id})
    cluster_datasets = list(cluster_series.values())

    # Malicious client ids for decode table colouring
    malicious_ids = set(Client.objects.filter(is_malicious=True)
                                      .values_list('client_id', flat=True))

    ctx = {
        'available_rounds': available_rounds,
        'action_result': action_result,
        'action_label': action_label,
        # stored results
        'decode_results': decode_results,
        'pca_latest': pca_latest,
        'clusters': clusters,
        'agg_results': agg_results,
        'malicious_ids': malicious_ids,
        # chart JSON
        'agg_rounds_json':    json.dumps(agg_rounds),
        'agg_acc_json':       json.dumps(agg_acc),
        'agg_loss_json':      json.dumps(agg_loss),
        'agg_clipped_json':   json.dumps(agg_clipped),
        'cluster_datasets_json': json.dumps(cluster_datasets),
        'has_decode':   decode_results.exists(),
        'has_clusters': clusters.exists(),
        'has_agg':      agg_results.exists(),
    }
    return render(request, 'core/fedgt_dashboard.html', ctx)
