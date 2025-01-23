from django import forms
from .models import Draw

class DrawForm(forms.ModelForm):
    class Meta:
        model = Draw
        fields = ['spade', 'heart', 'diamond', 'club']