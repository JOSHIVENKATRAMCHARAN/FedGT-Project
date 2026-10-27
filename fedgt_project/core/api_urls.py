from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import api_views

router = DefaultRouter()
router.register('clients', api_views.ClientViewSet)
router.register('client-data', api_views.ClientDataViewSet)
router.register('poisoned-clients', api_views.PoisonedClientViewSet)
router.register('groups', api_views.GroupAssignmentViewSet)
router.register('model-updates', api_views.ModelUpdateViewSet)
router.register('test-results', api_views.TestResultViewSet)
router.register('decode-results', api_views.FedGTDecodeResultViewSet)
router.register('pca-scores', api_views.PCAScoreViewSet)
router.register('clusters', api_views.ClientClusterViewSet)
router.register('aggregation-results', api_views.AggregationResultViewSet)
router.register('workflow', api_views.FedGTWorkflowViewSet, basename='workflow')

urlpatterns = [
    path('', include(router.urls)),
]
