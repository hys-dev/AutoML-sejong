from django import forms
from sejong.models import *

class ImageNasForm(forms.ModelForm):
    class Meta:
        model = ImageNas
        fields = []

class MultimodalNasForm(forms.ModelForm):
    class Meta:
        model = MultimodalNas
        fields = []