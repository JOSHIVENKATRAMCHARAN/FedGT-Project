from django.db import models


class Client(models.Model):
    client_id = models.CharField(max_length=20, unique=True)
    client_name = models.CharField(max_length=100)
    region = models.CharField(max_length=100)
    device_type = models.CharField(max_length=100)
    is_malicious = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.client_id} - {self.client_name}"


class ClientData(models.Model):
    sample_id = models.CharField(max_length=20)
    client_id = models.CharField(max_length=20)
    feature_1 = models.FloatField()
    feature_2 = models.FloatField()
    feature_3 = models.FloatField()
    feature_4 = models.FloatField()
    feature_5 = models.FloatField()
    feature_6 = models.FloatField()
    feature_7 = models.FloatField()
    feature_8 = models.FloatField()
    feature_9 = models.FloatField()
    feature_10 = models.FloatField()
    label = models.IntegerField()


class PoisonedClient(models.Model):
    client_id = models.CharField(max_length=20)
    attack_type = models.CharField(max_length=50)
    source_label = models.IntegerField()
    target_label = models.IntegerField()
    attack_strength = models.FloatField()


class GroupAssignment(models.Model):
    group_id = models.CharField(max_length=20)
    client_id = models.CharField(max_length=20)


class ModelUpdate(models.Model):
    round = models.IntegerField()
    client_id = models.CharField(max_length=20)
    loss = models.FloatField()
    accuracy = models.FloatField()
    update_norm = models.FloatField()
    gradient_variance = models.FloatField()


class TestResult(models.Model):
    round = models.IntegerField()
    group_id = models.CharField(max_length=20)
    test_result = models.IntegerField()
    utility_score = models.FloatField()
    pca_score = models.FloatField()


class TrainedModel(models.Model):
    MODEL_CHOICES = [
        ('random_forest', 'Random Forest'),
        ('logistic_regression', 'Logistic Regression'),
        ('svm', 'Support Vector Machine'),
        ('knn', 'K-Nearest Neighbors'),
    ]
    name = models.CharField(max_length=50, choices=MODEL_CHOICES)
    accuracy = models.FloatField(default=0)
    precision = models.FloatField(default=0)
    recall = models.FloatField(default=0)
    f1_score = models.FloatField(default=0)
    trained_at = models.DateTimeField(auto_now=True)
    confusion_matrix_json = models.TextField(default='[]')

    def __str__(self):
        return f"{self.name} (Acc: {self.accuracy:.2f})"


# ─── FedGT Engine Models ──────────────────────────────────────────────────────

class FedGTDecodeResult(models.Model):
    """Stores suspicion scores computed by the FedGT decoding algorithm."""
    client_id = models.CharField(max_length=20)
    round_number = models.IntegerField(default=0)   # 0 = aggregated across all rounds
    suspicion_score = models.FloatField(default=0.0)
    flagged = models.BooleanField(default=False)
    computed_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('client_id', 'round_number')

    def __str__(self):
        return f"{self.client_id} — score={self.suspicion_score:.4f} flagged={self.flagged}"


class PCAScore(models.Model):
    """Stores PCA-based anomaly scores computed live from ModelUpdate features."""
    client_id = models.CharField(max_length=20)
    round = models.IntegerField()
    score = models.FloatField()
    computed_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('client_id', 'round')

    def __str__(self):
        return f"{self.client_id} R{self.round} PCA={self.score:.4f}"


class ClientCluster(models.Model):
    """Stores K-Means cluster assignment and 2D PCA coordinates per client."""
    client_id = models.CharField(max_length=20, unique=True)
    cluster_label = models.IntegerField()
    pca_x = models.FloatField()
    pca_y = models.FloatField()
    computed_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.client_id} → Cluster {self.cluster_label}"


class AggregationResult(models.Model):
    """Stores the result of secure FedAvg aggregation per round."""
    round = models.IntegerField(unique=True)
    global_accuracy = models.FloatField()
    global_loss = models.FloatField()
    excluded_clients_json = models.TextField(default='[]')
    clipped_count = models.IntegerField(default=0)
    included_count = models.IntegerField(default=0)
    computed_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Round {self.round} — GlobalAcc={self.global_accuracy:.4f}"

    def excluded_clients_list(self):
        import json
        return json.loads(self.excluded_clients_json)
