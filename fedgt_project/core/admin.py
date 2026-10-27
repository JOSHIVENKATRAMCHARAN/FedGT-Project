from django.contrib import admin

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
    TrainedModel,
)


admin.site.register(Client)
admin.site.register(ClientData)
admin.site.register(PoisonedClient)
admin.site.register(GroupAssignment)
admin.site.register(ModelUpdate)
admin.site.register(TestResult)
admin.site.register(TrainedModel)
admin.site.register(FedGTDecodeResult)
admin.site.register(PCAScore)
admin.site.register(ClientCluster)
admin.site.register(AggregationResult)
