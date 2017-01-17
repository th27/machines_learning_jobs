from flask import Flask, render_template, jsonify, request
import pygal
from pygal.style import LightenStyle
from pygal.style import Style
from pygal import Config

import pandas as pd
import numpy as np
from scipy import spatial

from bs4 import BeautifulSoup
import requests

import re
import pickle

from nltk.corpus import stopwords, treebank
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, wordpunct_tokenize, WhitespaceTokenizer

from textblob import TextBlob

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn import feature_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances

from sklearn.linear_model import LogisticRegression

'==================================================================================================='

# place to unpickle everything you need

# tfidf_corpus vectorizer
with open('tfidf_corpus.pkl','rb') as p:
    tfidf_corpus = pickle.load(p)
    
# first model
with open('first_model.pkl','rb') as p:
    model = pickle.load(p)
    
# the DF with the canon
with open('DF_train_canon.pkl','rb') as p:
    DF_train_canon = pickle.load(p)
    
# second MODEL
with open('MODEL.pkl','rb') as p:
    MODEL = pickle.load(p)

# job type to automation rate pickle
with open('title_automation_dict.pkl','rb') as p:
    automation_dict = pickle.load(p)

# job type to automation rate pickle
with open('df_dist.pkl','rb') as p:
    df_dist = pickle.load(p)

# protected skill matrix
with open('df_scaled.pkl','rb') as p:
    df_scaled = pickle.load(p)

'==================================================================================================='

# initialize variables

n = 5
desc = ''
tokens = []
predicted_class = ''
class_confidence = ''
attributes = []


'==================================================================================================='

# define all relevant functions here

def url_to_desc(url):
    '''scrapes indeed url link for job description text'''
    soup = BeautifulSoup(requests.get(url).text,'lxml')
    doc = soup.find('span',class_='summary').get_text(separator=' ')
    return doc

def clean_stop_stem(doc):
    '''
    converts raw job description texts into string, cleaned, stop-words applied and stemmed
    '''
    # convert to string
    doc  = str(doc)

    # replace 'n\' with spaces
    doc = doc.replace('\n',' ')

    # remove list of named entities (single words only)
    # only do for location and people, not for general pronouns
    
    # make it text only
    doc = re.sub('[^a-zA-Z]',' ',doc)

    # tokenize by whitespace and lower case
    words = TextBlob(doc).words.lower()

    # remove stopwords
    # move this into a csv
    selected_words = ['aa','aaa','aama','aanp','aap','ab','aba','equal','for','job','opportunity',]
    stoppers = set(stopwords.words('english') + list(selected_words) + list(ENGLISH_STOP_WORDS))
    stoppers = list(stoppers)
    words = [i for i in words if i not in stoppers]

    # # stem only
    stemmer = PorterStemmer()
    stem_tokens = [stemmer.stem(word) for word in words]

    return ' '.join(stem_tokens)

def zipped_list(prob_dist):
    zipped_list_single = []
    for i in range(len(prob_dist)):
        zipped_single = list(zip(prob_dist[i],model.classes_))
        sorted_zipped_single = sorted(zipped_single,reverse=True)
        top_n_single = sorted_zipped_single[:n]
        zipped_list_single.append(top_n_single)
        return zipped_list_single

def cosine_similarity(vector_1, vector_2):
    '''
    for vectors of the same size, return an array that is the cosine similarity between two vectors
    '''
    cos_sim_single = []

    for i in range(len(vector_1)):
        doc_single = vector_1.iloc[i].reshape(1,-1)
        can_single = vector_2.iloc[i].reshape(1,-1)
        cos_sim_single.append(1 - spatial.distance.cosine(doc_single, can_single))
    
    return cos_sim_single

def tokens_to_doc_vecs_df(tokens):
    '''
    takes input doc tokens and then converts into a doc_vecs_single_df
    '''    
    # convert tokens into sparse matrix using tfidf_corpus
    doc_vecs_single = tfidf_corpus.transform([tokens])
    
    # Store them in a Pandas DataFrame
    doc_vecs_single_df = pd.DataFrame(doc_vecs_single.todense(), columns=[tfidf_corpus.get_feature_names()])
    
    return doc_vecs_single_df


def features_for_MODEL(tokens, vec_df):
    '''
    takes input doc tokens and then converts into a feature for MODEL
    '''
    # generate predict_proba for single set
    # model_predicted_classes = model.classes_
    model_probas_single = model.predict_proba(vec_df)
    
    # break out model_proba
    zipped_list_single = zipped_list(model_probas_single)

    # flatten and split into two lists
    flat_zipped_list_single = [item for sublist in zipped_list_single for item in sublist]
    flat_zipped_list_single_0 = [i[0] for i in flat_zipped_list_single]
    flat_zipped_list_single_1 = [i[1] for i in flat_zipped_list_single]
    
    # add original texts
    X_single_repeat = [tokens] * n
    
    # zip it all together and store in a DF
    model_single_zipped = list(zip(flat_zipped_list_single_0,flat_zipped_list_single_1,X_single_repeat))
    DF_single = pd.DataFrame(model_single_zipped)
   
    # rename columns
    DF_single.columns = ['y_single_pred_probas','y_single_pred','X_single']
    
    # concatenate the canonical text
    DF_single_combined = pd.merge(DF_single,DF_train_canon,left_on='y_single_pred',right_on='y_train',how='left')
    
    # drop unnecessary columns from DF_combined
    DF_single_combined_1 = DF_single_combined.drop(['y_train'],axis=1)

    # rename columns
    cols = ['y_single_pred_probas', 'y_single_pred', 'X_single', 'class_canon_single']
    DF_single_combined_1.columns = cols

    # vectorize the extended X_single with tfidf_corpus
    # vectorize the X_single with tfidf_corpus
    doc_vecs_single_extend = tfidf_corpus.transform(DF_single_combined_1.X_single)

    # Store them in a Pandas DataFrame
    doc_vecs_single_extend_df = pd.DataFrame(doc_vecs_single_extend.todense(), columns=[tfidf_corpus.get_feature_names()])

    # vectorize the extended class_canon_text with tfidf_corpus
    # vectorize the canon_single with tfidf_corpus
    doc_vecs_canon_single = tfidf_corpus.transform(DF_single_combined_1.class_canon_single)

    # Store them in a Pandas DataFrame
    doc_vecs_canon_single_df = pd.DataFrame(doc_vecs_canon_single.todense(), columns=[tfidf_corpus.get_feature_names()])
    
    # create new set of features which is the difference between doc vectors and canon vectors
    doc_vecs_diff_single_df = doc_vecs_single_extend_df - doc_vecs_canon_single_df
    
    DF_single_combined_1['cos_sim_single'] = cosine_similarity(doc_vecs_single_extend_df,doc_vecs_canon_single_df)
    
    FEATURES_single = pd.concat([doc_vecs_single_extend_df, \
                                   doc_vecs_canon_single_df, \
                                   doc_vecs_diff_single_df, \
                                   DF_single_combined_1.y_single_pred_probas, \
                                   DF_single_combined_1.cos_sim_single],axis=1)
    
    return FEATURES_single

def predict_class(features,doc_vecs_single_df):
    predict_proba_MODEL = MODEL.predict_proba(features)
    predict_proba_final = predict_proba_MODEL[:,1]
    job_cats = [i[1] for i in zipped_list(model.predict_proba(doc_vecs_single_df))[0]]
    jobcat_class = list(zip(predict_proba_final,job_cats))
    predicted_class = sorted(jobcat_class,reverse=True)[0][1]
    return predicted_class

def url_to_predict(url):
    desc = url_to_desc(url)
    tokens = clean_stop_stem(desc)
    doc_vecs_single_df = tokens_to_doc_vecs_df(tokens)
    FEATURES = features_for_MODEL(tokens,doc_vecs_single_df)
    predicted_class = predict_class(FEATURES,doc_vecs_single_df)
    return predicted_class

def class_to_automation(predicted_class):
	automation_rate = automation_dict[predicted_class]
	return automation_rate

def job_to_top_5(job_type):
    '''
    take job and return the five closest jobs with lowest automation risk
    '''
    
    capture = 7
    n_return = 3
    top_n_list = []
    lower_ar_jobs = []
    
    top_n = df_dist.loc[:,job_type].nlargest(capture)
    for i in top_n.index:
        top_n_list.append((automation_dict[i],i))
            
    for j in top_n_list:
        if j[0] < automation_dict[job_type]:
            lower_ar_jobs.append(j)
        sorted_lowest = sorted(lower_ar_jobs)[:n_return]
    
    return sorted_lowest

def protected_attributes(job_type):
    return df_scaled.loc[job_type,:]

'==================================================================================================='

app = Flask(__name__)

@app.route('/')
def my_form():
	return render_template('my_form.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':

		global n, predicted_class

		url = request.form["url"]
		predicted_class = str(url_to_predict(url))
		automation_rate = class_to_automation(predicted_class)
		attributes = [i for i in protected_attributes(predicted_class)]
		top_5_better = [i for i in job_to_top_5(predicted_class)]

		automation_score = "{0:.0f}%".format(automation_rate * 100)

		num_1_job = top_5_better[0][1]
		num_1_rate = top_5_better[0][0]
		num_2_job = top_5_better[1][1]
		num_2_rate = top_5_better[1][0]
		num_3_job = top_5_better[2][1]
		num_3_rate = top_5_better[2][0]

		attributes_1 = [i for i in protected_attributes(num_1_job)]
		attributes_2 = [i for i in protected_attributes(num_2_job)]
		attributes_3 = [i for i in protected_attributes(num_3_job)]

		# section 1
		gauge_style = Style(background='transparent',
							 plot_background='transparent',
						     # foreground='#53E89B',
						     # foreground_strong='#53A0E8',
						     # foreground_subtle='#630C0D',
						     opacity='.6',
						     opacity_hover='.9',
						     transition='4000ms'
						     )

		gauge = pygal.SolidGauge(inner_radius=0.55,	
								 show_legend=False,
								 width=300,
								 height=300,
								 margin=0,
								 title=False)	
		percent_formatter = lambda x: '{:.10g}%'.format(x)
		gauge.value_formatter = percent_formatter

		gauge.add('', [{'value': automation_rate * 100, 
						'max_value': 100, 
						'color': '#F44336',
						}])
		gauge_data = gauge.render_data_uri()

		# section 2
		# 2a
		gauge_1 = pygal.SolidGauge(inner_radius=0.66,	
								   show_legend=False,
								   width=150,
								   height=150,
								   margin=0,
								   title=False,
								   style=gauge_style)	
		percent_formatter = lambda x: '{:.10g}%'.format(x)
		gauge_1.value_formatter = percent_formatter

		gauge_1.add('', [{'value': num_1_rate * 100, 
						'max_value': 100, 
						'color': '#3F51B5'
						}])
		gauge_1_data = gauge_1.render_data_uri()

		# 2b
		gauge_2 = pygal.SolidGauge(inner_radius=0.66,	
								   show_legend=False,
								   width=150,
								   height=150,
								   margin=0,
								   title=False,
								   style=gauge_style)	
		percent_formatter = lambda x: '{:.10g}%'.format(x)
		gauge_2.value_formatter = percent_formatter
	
		gauge_2.add('', [{'value': num_2_rate * 100, 
						'max_value': 100, 
						'color': '#009688',
						}])
		gauge_2_data = gauge_2.render_data_uri()

		# 2c
		gauge_3 = pygal.SolidGauge(inner_radius=0.66,	
								   show_legend=False,
								   width=150,
								   height=150,
								   margin=0,
								   title=False,
								   style=gauge_style)	
		percent_formatter = lambda x: '{:.10g}%'.format(x)
		gauge_3.value_formatter = percent_formatter

		gauge_3.add('', [{'value': num_3_rate * 100, 
						'max_value': 100, 
						'color': '#FFC107',
						}])
		gauge_3_data = gauge_3.render_data_uri()

		# section 3

		radar_style = Style(background='transparent',
							  plot_background='transparent',
							  font_family='Roboto',
							  # label_font_family='Roboto',
							  # label_font_size='20px',
							  # major_label_font_size='32px',
							  # colors='green','blue','orange',
							  # transition='400ms ease-in'
							  )

		radar = pygal.Radar(title=False,
							style=radar_style,
							width=700,
							height=500, 
							margin=40,
							)
		radar.x_labels = ["Originality","Manual dexterity","Finger dexterity","Social perceptiveness", \
						 "Persuasion skills", "Negotiation skills", "Cramped environment","Fine arts"," Human assistance"]
		
		# radar.add(predicted_class, attributes,[{'color':'orange'}])
		# radar.add(num_1_job, attributes_1,[{'color':'#008000'}])
		# radar.add(num_2_job, attributes_2,[{'color':'#329932'}])
		# radar.add(num_3_job, attributes_3,[{'color':'#84C184'}])

		radar.add(predicted_class, attributes,stroke_style={'width': 1},fill=True)
		radar.add(num_1_job, attributes_1,stroke_style={'width': 4, 'linecap': 'round', 'linejoin': 'round'})
		radar.add(num_2_job, attributes_2,stroke_style={'width': 4, 'linecap': 'round', 'linejoin': 'round'})
		radar.add(num_3_job, attributes_3,stroke_style={'width': 4, 'linecap': 'round', 'linejoin': 'round'})

		radar_data = radar.render_data_uri()

	return render_template("result.html",url=url,
										predicted_class=predicted_class,
										automation_rate=automation_rate,
										automation_score=automation_score, 
										attributes=attributes,
										top_5_better=top_5_better, 
										gauge=gauge_data,
										num_1_job=num_1_job, 
										num_2_job=num_2_job, 
										num_3_job=num_3_job, 
										gauge_1 = gauge_1_data,
										gauge_2 = gauge_2_data,
										gauge_3 = gauge_3_data,
										radar=radar_data)

if __name__ == '__main__':
    app.run(debug=True)
