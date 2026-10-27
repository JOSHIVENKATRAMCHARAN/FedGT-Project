import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_data(base_dir=None, replace=False):
    from .models import (Client, ClientData, PoisonedClient,
                         GroupAssignment, ModelUpdate, TestResult)

    data_dir = base_dir or BASE_DIR

    # Clients
    if replace:
        TestResult.objects.all().delete()
        ModelUpdate.objects.all().delete()
        GroupAssignment.objects.all().delete()
        PoisonedClient.objects.all().delete()
        ClientData.objects.all().delete()
        Client.objects.all().delete()

    if not Client.objects.exists():
        df = pd.read_csv(os.path.join(data_dir, 'clients.csv'))
        objs = [Client(client_id=r.client_id, client_name=r.client_name,
                       region=r.region, device_type=r.device_type,
                       is_malicious=bool(r.is_malicious)) for _, r in df.iterrows()]
        Client.objects.bulk_create(objs)

    # ClientData
    if not ClientData.objects.exists():
        df = pd.read_csv(os.path.join(data_dir, 'client_data.csv'))
        objs = [ClientData(
            sample_id=r.sample_id, client_id=r.client_id,
            feature_1=r.feature_1, feature_2=r.feature_2, feature_3=r.feature_3,
            feature_4=r.feature_4, feature_5=r.feature_5, feature_6=r.feature_6,
            feature_7=r.feature_7, feature_8=r.feature_8, feature_9=r.feature_9,
            feature_10=r.feature_10, label=r.label
        ) for _, r in df.iterrows()]
        ClientData.objects.bulk_create(objs, batch_size=500)

    # PoisonedClients
    if not PoisonedClient.objects.exists():
        df = pd.read_csv(os.path.join(data_dir, 'poisoned_clients.csv'))
        objs = [PoisonedClient(
            client_id=r.client_id, attack_type=r.attack_type,
            source_label=r.source_label, target_label=r.target_label,
            attack_strength=r.attack_strength
        ) for _, r in df.iterrows()]
        PoisonedClient.objects.bulk_create(objs)

    # GroupAssignments
    if not GroupAssignment.objects.exists():
        df = pd.read_csv(os.path.join(data_dir, 'group_assignments.csv'))
        objs = [GroupAssignment(group_id=r.group_id, client_id=r.client_id)
                for _, r in df.iterrows()]
        GroupAssignment.objects.bulk_create(objs)

    # ModelUpdates
    if not ModelUpdate.objects.exists():
        df = pd.read_csv(os.path.join(data_dir, 'model_updates.csv'))
        objs = [ModelUpdate(
            round=r['round'], client_id=r.client_id,
            loss=r.loss, accuracy=r.accuracy,
            update_norm=r.update_norm, gradient_variance=r.gradient_variance
        ) for _, r in df.iterrows()]
        ModelUpdate.objects.bulk_create(objs)

    # TestResults
    if not TestResult.objects.exists():
        df = pd.read_csv(os.path.join(data_dir, 'test_results.csv'))
        objs = [TestResult(
            round=r['round'], group_id=r.group_id,
            test_result=r.test_result,
            utility_score=r.utility_score, pca_score=r.pca_score
        ) for _, r in df.iterrows()]
        TestResult.objects.bulk_create(objs)

    return True
