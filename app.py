from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB

clf = pickle.load(open("Sent.pickle", 'rb'))
cv=pickle.load(open('transform.pickle','rb'))
app = Flask(__name__)

@app.route('/')
def home():
		return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
	message = ''
	prediction = ''
	if request.method == 'POST':
			message = request.form['message']
	
	data = [message]
	print(data)
	vect = cv.transform(data).toarray()
	print(vect)
	prediction = clf.predict(vect)
	print(prediction)
	return render_template('result.html',pred =prediction)

if __name__ == '__main__':
	app.run(debug=True)
