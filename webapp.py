from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

from recurrent_module import eval_char_model
 
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
 
class ReusableForm(Form):
    name = TextField('test_string:', validators=[validators.required()])
 
 
@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
 
    print form.errors
    if request.method == 'POST':
        input_string=request.form['name']
 
        if form.validate():
            # Save the comment here.
            flash('Your reconstruction loss/bpc %f' % eval_char_model(input_string))
        else:
            flash('Please Enter a phrase to reconstruct. ')
 
    return render_template('evaluates.html', form=form)
 
if __name__ == "__main__":
    app.run()
