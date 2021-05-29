from django import forms

class LoginForm(forms.Form):
	username = forms.CharField(
		required = True,
		label = 'username',
		max_length = 32
	)
	password = forms.CharField(
		required = True,
		label = 'password',
		max_length = 32,
		widget = forms.PasswordInput()
	)
	
class SentimentForm(forms.Form):
	message = forms.CharField(
		required = True,
		label = 'username',
		max_length = 1000
	)

class UserRegistrationForm(forms.Form):
	username = forms.CharField(
		required = True,
		label = 'username',
		max_length = 32
	)
	email = forms.CharField(
		required = True,
		label = 'email',
		max_length = 32,
	)
	password = forms.CharField(
		required = True,
		label = 'password',
		max_length = 32,
		widget = forms.PasswordInput()
	)