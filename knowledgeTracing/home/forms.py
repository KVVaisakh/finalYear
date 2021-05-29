from django import forms

class ModelFormWithFileField(forms.Form):
	file = forms.FileField(
			required = True,
			label = 'file',
		)