# This is basically the heart of my flask 

from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


from model import recommendation_model

app = Flask(__name__)  # intitialize the flaks app  # common 


def cleaning():
    print("test")

@app.route('/', methods =["GET", "POST"])
def test():
    if request.method == "POST":
            # getting input with name = fname in HTML form
        first_name = request.form.get("fname")
        print('Input from html : ' + str(first_name))
        top_5 =recommendation_model(first_name)



        return  render_template('index.html',tables=[top_5.to_html(classes='age')], titles = ['NAN', 'Age Prediction'])
    
    return render_template("index.html")


# Any HTML template in Flask App render_template

if __name__ == '__main__' :
    app.run(debug=True )  # this command will enable the run of your flask app or api
    
    #,host="0.0.0.0")





