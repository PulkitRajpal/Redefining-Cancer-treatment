from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 

# ML Packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	df= pd.read_csv("C:/Users/pulki/Desktop/hackathon/data/mydata.csv")
	# Features and Labels
	df_X = df.Text1
	df_Y = df.Class

	# def decontracted(phrase):
 #    	phrase = re.sub(r"won\'t", "will not", phrase)
 #    	phrase = re.sub(r"can\'t", "can not", phrase)

    
 #    	phrase = re.sub(r"n\'t", " not", phrase)
 #    	phrase = re.sub(r"\'re", " are", phrase)
 #    	phrase = re.sub(r"\'s", " is", phrase)
 #    	phrase = re.sub(r"\'d", " would", phrase)
 #    	phrase = re.sub(r"\'ll", " will", phrase)
 #    	phrase = re.sub(r"\'t", " not", phrase)
 #    	phrase = re.sub(r"\'ve", " have", phrase)
 #    	phrase = re.sub(r"\'m", " am", phrase)
 #    	return phrase

	# stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
 #            	"you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
 #            	'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
 #            	'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
 #            	'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
 #            	'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
 #            	'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
 #            	'above', 'below', 'to', 'from', 'up', 'down', 'in', 'on', 'off', 'over', 'under', 'again', 'further',\
 #            	'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
 #            	'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', \
 #            	's', 't'])
 #    updated_text= []



	# for sentance in (df_X.values):
 #    	sentance = re.sub(r"http/S+", "", sentance)
 #    	sentance = BeautifulSoup(sentance, 'lxml').get_text()
 #    	sentance = decontracted(sentance)
 #    	sentance = re.sub("/S*/d/S*", "", sentance).strip()
 #    	sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    
 #    	sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
 #    	updated_text.append(sentance.strip())
    
    # Vectorization
	corpus = df_X
	cv = TfidfVectorizer(ngram_range=(1,2), min_df=10)
	X = cv.fit_transform(corpus) 
	
	# Loading our ML Model
	naivebayes_model = open("models/system_naive.pkl","rb")
	clf = joblib.load(naivebayes_model)

	# Receives the input query from form
	if request.method == 'POST':
		namequery = request.form['namequery']
		data = [namequery]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('results.html',prediction = my_prediction,name = namequery.upper())


if __name__ == '__main__':
	app.run(debug=True)