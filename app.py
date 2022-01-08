from flask import Flask ,render_template,request,jsonify,session
from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import sqlite3 as sql
import base64
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from flask_bootstrap import Bootstrap
import numpy as np
from sklearn.utils import shuffle
import os
from flask import Flask, render_template, request, url_for,send_from_directory
import os
#from geo import getTweetLocation


app = Flask(__name__)
app.secret_key = 'any random string'
PEOPLE_FOLDER = os.path.join('static', 'people_photo')
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

   
def validate(username,password):
    con = sql.connect('static/chat.db')
    completion = False
    with con:
        cur = con.cursor()
        cur.execute('SELECT * FROM persons')
        rows = cur.fetchall()
        for row in rows:
            dbuser = row[1]
            dbpass = row[2]
            if dbuser == username:
                completion = (dbpass == password)
    return completion


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        completion = validate(username,password)
        if completion == False:
            error = 'invalid Credentials. please try again.'
        else:
            session['username'] = request.form['username']
            return render_template('index.html')
    return render_template('search.html', error=error)



    
@app.route('/register', methods = ['GET','POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            username = request.form['username']
            password = request.form['password']
            with sql.connect("static/chat.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO persons(name,username,password) VALUES (?,?,?)",(name,username,password))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"
        finally:
            return render_template("index.html",msg = msg)
            con.close()
    return render_template('register.html')


@app.route('/list')
def list():
   con = sql.connect("static/chat.db")
   con.row_factory = sql.Row
   
   cur = con.cursor()
   cur.execute("select * from persons")
   
   rows = cur.fetchall();
   return render_template("list.html",rows = rows)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


@app.route('/predict', methods=['POST'])
def predict():
   df = pd.read_csv("spam.csv", encoding="latin-1")
   #df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
   # Features and Labels
   df['label'] = df['class'].map({'ham': 0, 'spam': 1})
   X = df['message']
   y = df['label']

   # Extract Feature With CountVectorizer
   cv = CountVectorizer()
   X = cv.fit_transform(X)  # Fit the Data
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
   # Naive Bayes Classifier
   from sklearn.naive_bayes import MultinomialNB

   clf = MultinomialNB()
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)
   # Alternative Usage of Saved Model
   # joblib.dump(clf, 'NB_spam_model.pkl')
   # NB_spam_model = open('NB_spam_model.pkl','rb')
   # clf = joblib.load(NB_spam_model)

   if request.method == 'POST':
      message = request.form['message']
      data = [message]
      vect = cv.transform(data).toarray()
      my_prediction = clf.predict(vect)

      if int(my_prediction)==1:
            prediction='Spam'
            return render_template('result2.html', prediction=prediction) 
      else:
            prediction='Not a Spam (It is a Ham)'
        
            return render_template("result3.html",prediction=prediction)




if __name__ == '__main__':
   app.run(debug = True )
