from django.urls import path
from .views import predict, import_excel_data

urlpatterns = [
    path('', predict, name='home'),
    # path('add_draw/', views.add_draw, name='add_draw'),
    path('import_excel/', import_excel_data, name="import_excel"),
]