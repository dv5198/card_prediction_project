from django import forms

class DrawForm(forms.Form):
    spade = forms.CharField(max_length=2)
    heart = forms.CharField(max_length=2)
    diamond = forms.CharField(max_length=2)
    club = forms.CharField(max_length=2)
