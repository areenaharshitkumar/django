"""
Definition of views.
"""

from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest
from django.http import HttpResponse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from google.cloud import translate_v2 as translate
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy
from langdetect import detect
from sklearn import preprocessing
from sklearn.cluster import KMeans
import re
import scipy
import nltk
import json
import xlwt
import xlrd
import os
import numpy as np
import socket
model = {}

def home(request):
    """Renders the home page."""
    
    if request.method == "POST":
        
        action = request.POST.get('form_data[action]')
        if action == "likedislike":
            last_row = 0
            hostname = socket.gethostname()
            IPAddr = socket.gethostbyname(hostname)
            wb = xlrd.open_workbook('IpAddress.xls')
            sheet = wb.sheet_by_index(0)
            f=0
            for i in range(1, sheet.nrows):
                if IPAddr == sheet.row_values(i)[1]:
                    f=1
                    break
            #print(sheet.cell_value(0, 1))
            if i == 0:
                last_row = sheet.nrows+1
                #wb = Workbook()
                #sheet1 = wb.add_sheet('Sheet 1')
            
                #sheet.write(last_row, 0, hostname)
                sheet.write(last_row, 1, request.POST.get('form_data[sid]'))
                wb.save('IpAddress.xls')
            
            return HttpResponse(last_row)
        
        if action == "emailid":
            last_row = 0
            wb = xlrd.open_workbook('emails.xls')
            sheet = wb.sheet_by_index(0)
            last_row = sheet.nrows+1
            #wb = Workbook()
            #sheet1 = wb.add_sheet('Sheet 1')
            
            sheet.write(last_row, 0, request.POST.get('form_data[text]'))
            wb.save('emails.xls')
            return HttpResponse('done')
        if action == "datarequest":
            global model
            t_data = ''
            show_out = ''
            source = ''
            link = ''
            d_lang ='en'
            try:
                wpt = nltk.WordPunctTokenizer()
                stop_words = nltk.corpus.stopwords.words('english')

                gloveFile='app/static/file/glove.6B.300d.txt'

           
                s2 = request.POST.get('form_data[text]') 
                if (detect(s2) != 'en'):
                    d_lang = detect(s2)
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "app/static/app/clientfile/clienttranslateapi.json"
                    translate_client = translate.Client()
                    translation = translate_client.translate(s2, target_language=detect(s2))
                    z = format(translation['translatedText'])
                    print(z)
                    translation = translate_client.translate(z, target_language='en')
                    s2 = format(translation['translatedText'])
                def loadGloveModel(gloveFile):
                    print("Loading Glove Model")
                    f = open(gloveFile, 'r', encoding='utf-8')
                    model = {}
                    for line in f:
                        splitLine = line.split()
                        word = splitLine[0]
                        embedding = np.array([float(val) for val in splitLine[1:]])
                        model[word] = embedding
                    print("Done.", len(model), " words loaded!")
                    return model
                if not bool(model):
                    model = loadGloveModel(gloveFile)

                def normalize_document(doc):
                    doc = re.sub(r'[^a-zA-Z\s]', ' ', doc, re.I | re.A)
                    doc = doc.lower()
                    doc = doc.strip()
                    tokens = wpt.tokenize(doc)
                    filtered_tokens = [token for token in tokens if token not in stop_words]
                    return filtered_tokens
                wb = xlrd.open_workbook('app/static/file/CoronaApp.xlsx')
                sheet = wb.sheet_by_index(0)
                max_per = 0
            
                sheet.cell_value(0, 0)
                for i in range(1, sheet.nrows):
                    try:
                        z = sheet.row_values(i)[0]
                        if (detect(sheet.row_values(i)[0]) != 'en'):
                            d_lang = detect(sheet.row_values(i)[0])
                            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "app/static/app/clientfile/clienttranslateapi.json"
                            translate_client = translate.Client()
                            translation = translate_client.translate(sheet.row_values(i)[0], target_language='en')
                            z = str(translation['translatedText'])
                        z = str(z)
                        ssf1 = normalize_document(z)
                        ssf2 = normalize_document(s2)
                        def cosine_distance_wordembedding_method(t1, t2):
                            vector_1 = np.mean([model[corpus] for corpus in t1], axis=0)
                            vector_2 = np.mean([model[corpus] for corpus in t2], axis=0)
                            cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
                            return (round((1 - cosine) * 100, 2))
                        t = cosine_distance_wordembedding_method(ssf2, ssf1)
                        if max_per < t:
                            max_per = t
                            show_out = sheet.row_values(i)[1]
                            source = sheet.row_values(i)[2]
                            link = sheet.row_values(i)[3]
                    except Exception as a:
                        f = open('error.txt', 'a+')
                        f.write('Line-'+str(i)+str(a) + ',')
                        pass

                if max_per > 60:
                    print(max_per)
                    t_data = show_out
                    if (detect(s2) == 'en') and (detect(t_data) != 'en'):
                        translation = translate_client.translate(t_data, target_language='en')
                        t_data = format(translation['translatedText'])


                else:
                    t_data = "We do not have enough information. Our team will get back to you with facts in next 24 hours"
                    if detect(s2) != 'en':
                        translation = translate_client.translate(t_data, target_language=d_lang)
                        t_data = format(translation['translatedText'])
                    source = ''
                    link = ''

            except Exception as e:
                print(e)
                t_data = 'Error'
        
            data = {'t_data': t_data,'source': source ,'link': link }
            dump = json.dumps(data)
            return HttpResponse(dump, content_type='application/json')
    
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
            
        }
    )

def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'Your contact page.',
            'year':datetime.now().year,
        }
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'Your application description page.',
            'year':datetime.now().year,
        }
    )

