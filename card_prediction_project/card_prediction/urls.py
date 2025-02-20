from django.urls import path
from .views import predict, import_excel_data,save_predictions

urlpatterns = [
    path('', predict, name='home'),
    # path('add_draw/', views.add_draw, name='add_draw'),
    path('import_excel/', import_excel_data, name="import_excel"),
    path('save_predictions/', save_predictions, name='save_predictions'),
]
