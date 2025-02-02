from django.db import models

class Draw(models.Model):
    id = models.AutoField(primary_key=True)
    Club = models.CharField(max_length=2)
    Diamond = models.CharField(max_length=2)
    Heart = models.CharField(max_length=2)
    spade = models.CharField(max_length=2)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Spade: {self.spade}, Heart: {self.Heart}, Diamond: {self.Diamond}, Club: {self.Club}"
