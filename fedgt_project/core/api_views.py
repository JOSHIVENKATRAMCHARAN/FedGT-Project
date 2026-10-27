from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from training.pytorch_engine import run_local_training

from .fedgt_engine import (
    cluster_clients,
    compute_pca_scores,
    decode_malicious_clients,
    run_secure_aggregation,
    run_secure_aggregation_all_rounds,
)
from .models import (
    AggregationResult,
    Client,
    ClientData,
    ClientCluster,
    FedGTDecodeResult,
    GroupAssignment,
    ModelUpdate,
    PCAScore,
    PoisonedClient,
    TestResult,
)
from .serializers import (
    AggregationResultSerializer,
    ClientClusterSerializer,
    ClientDataSerializer,
    ClientSerializer,
    FedGTDecodeResultSerializer,
    GroupAssignmentSerializer,
    ModelUpdateSerializer,
    PCAScoreSerializer,
    PoisonedClientSerializer,
    TestResultSerializer,
)


class ClientViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Client.objects.all().order_by('client_id')
    serializer_class = ClientSerializer


class ClientDataViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ClientData.objects.all().order_by('sample_id')
    serializer_class = ClientDataSerializer


class PoisonedClientViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = PoisonedClient.objects.all().order_by('client_id')
    serializer_class = PoisonedClientSerializer


class GroupAssignmentViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = GroupAssignment.objects.all().order_by('group_id', 'client_id')
    serializer_class = GroupAssignmentSerializer


class ModelUpdateViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ModelUpdate.objects.all().order_by('round', 'client_id')
    serializer_class = ModelUpdateSerializer


class TestResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = TestResult.objects.all().order_by('round', 'group_id')
    serializer_class = TestResultSerializer


class FedGTDecodeResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = FedGTDecodeResult.objects.all().order_by('-suspicion_score')
    serializer_class = FedGTDecodeResultSerializer


class PCAScoreViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = PCAScore.objects.all().order_by('round', '-score')
    serializer_class = PCAScoreSerializer


class ClientClusterViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ClientCluster.objects.all().order_by('cluster_label', 'client_id')
    serializer_class = ClientClusterSerializer


class AggregationResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AggregationResult.objects.all().order_by('round')
    serializer_class = AggregationResultSerializer


class FedGTWorkflowViewSet(viewsets.ViewSet):
    """Action API for the FL controller, FedGT engine, and secure aggregator."""

    @action(detail=False, methods=['post'])
    def train(self, request):
        round_number = request.data.get('round_number')
        result = run_local_training(
            round_number=int(round_number) if round_number else None,
            epochs=int(request.data.get('epochs', 2)),
        )
        return Response(result)

    @action(detail=False, methods=['post'])
    def decode(self, request):
        round_number = request.data.get('round_number')
        return Response(decode_malicious_clients(round_number))

    @action(detail=False, methods=['post'])
    def pca(self, request):
        round_number = request.data.get('round_number')
        return Response(compute_pca_scores(round_number))

    @action(detail=False, methods=['post'])
    def cluster(self, request):
        n_clusters = int(request.data.get('n_clusters', 3))
        return Response(cluster_clients(n_clusters=n_clusters))

    @action(detail=False, methods=['post'])
    def aggregate(self, request):
        round_number = request.data.get('round_number')
        if round_number:
            return Response(run_secure_aggregation(int(round_number)))
        return Response(run_secure_aggregation_all_rounds())
