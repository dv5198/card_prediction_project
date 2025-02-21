from django.db import models


class Draw(models.Model):
    id = models.AutoField(primary_key=True)
    Club = models.CharField(max_length=2)
    Diamond = models.CharField(max_length=2)
    Heart = models.CharField(max_length=2)
    spade = models.CharField(max_length=2)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'Draw'  # 🔄 This changes the table name to just "Draw"
    def __str__(self):
        return f"Spade: {self.spade}, Heart: {self.Heart}, Diamond: {self.Diamond}, Club: {self.Club}"

class Prediction(models.Model):

    monte_carlo = models.JSONField(null=True, blank=True)  # ✅ Updated
    random_forest = models.JSONField(null=True, blank=True)
    lstm = models.JSONField(null=True, blank=True)
    trend_analysis = models.JSONField(null=True, blank=True)  # ✅ Updated
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'Prediction'  # 🔄 This changes the table name to just "Prediction"
    def __str__(self):
        return f"Prediction on {self.timestamp}"