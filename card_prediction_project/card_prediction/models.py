from django.db import models

class Draw(models.Model):
    spade = models.CharField(max_length=2)
    heart = models.CharField(max_length=2)
    diamond = models.CharField(max_length=2)
    club = models.CharField(max_length=2)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Spade: {self.spade}, Heart: {self.heart}, Diamond: {self.diamond}, Club: {self.club}"
