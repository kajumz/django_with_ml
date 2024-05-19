# mlapp/forms.py
from django import forms


class TextForm(forms.Form):
    name = forms.CharField(label='Your Name', max_length=100)
    email = forms.EmailField(label='Your Email')
    message = forms.CharField(widget=forms.Textarea, label='Your Message')


class InputForm(forms.Form):
    first_name = forms.CharField(max_length=200)
    last_name = forms.CharField(max_length=200)
    roll_number = forms.IntegerField(
        help_text="Enter 6 digit roll number"
    )
    password = forms.CharField(widget=forms.PasswordInput())
