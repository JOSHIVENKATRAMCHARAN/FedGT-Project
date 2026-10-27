from rest_framework import serializers

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


class ClientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Client
        fields = '__all__'


class ClientDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClientData
        fields = '__all__'


class PoisonedClientSerializer(serializers.ModelSerializer):
    class Meta:
        model = PoisonedClient
        fields = '__all__'


class GroupAssignmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = GroupAssignment
        fields = '__all__'


class ModelUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelUpdate
        fields = '__all__'


class TestResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = TestResult
        fields = '__all__'


class FedGTDecodeResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = FedGTDecodeResult
        fields = '__all__'


class PCAScoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = PCAScore
        fields = '__all__'


class ClientClusterSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClientCluster
        fields = '__all__'


class AggregationResultSerializer(serializers.ModelSerializer):
    excluded_clients = serializers.SerializerMethodField()

    class Meta:
        model = AggregationResult
        fields = '__all__'

    def get_excluded_clients(self, obj):
        return obj.excluded_clients_list()
