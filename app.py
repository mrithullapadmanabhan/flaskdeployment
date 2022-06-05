from __future__ import print_function
from flask import Flask,request, url_for, redirect, render_template
import pickle
import sys
import io
import pandas as pd
import re
import sklearn
from nltk.corpus import stopwords
import waitress
from pandas.io.formats import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from io import StringIO
import csv
from csv import writer
import logging
import numpy as np


app = Flask(__name__)


@app.route('/')
def hello_world():
    print('bhjdfhjd', file=sys.stderr)
    app.logger.info('testing info log')

    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    f = request.files['data_file']
    file_path = "./csv_files/" + f.filename
    f.save(file_path)
    if not f:
        return "No file"
    #f = request.files['data_file']
    #if not f:
    #    return "No file"

    from model import word_vectorizer
    model = pickle.load(open('model.pkl', 'rb'))
    # stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    # csv_input = csv.reader(stream)
    # # print("file contents: ", file_contents)
    # # print(type(file_contents))
    # print(csv_input)
    # i = 0
    # for row in csv_input:
    #     if (i == 1):
    #         print(row)
    #         with open('NewData.csv', 'a', newline='') as f_object:
    #             # Pass the CSV  file object to the writer() function
    #             writer_object = writer(f_object)
    #             # Result - a writer object
    #             # Pass the data in the list as an argument into the writerow() function
    #             writer_object.writerow(row)
    #             # Close the file object
    #             f_object.close()
    #     i+=1
    #


    res = pd.read_csv('./csv_files/test1.csv', encoding='utf-8')
    res['cleaned_resume'] = ''
    print(res.head())

    def cleanResume(resumeText):
        resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                            resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
        resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
        return resumeText

    res['cleaned_resume'] = res.Resume.apply(lambda x: cleanResume(x))

    res.head()

    new_res = res.copy()

    import nltk
    nltk.download('stopwords')
    import nltk
    nltk.download('punkt')
    import string
    oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"])
    totalWords = []
    new_Sentences = res['Resume'].values
    new_cleanedSentences = ""
    for records in new_Sentences:
        cleanedText = cleanResume(records)
        new_cleanedSentences += cleanedText
        requiredWords = nltk.word_tokenize(cleanedText)
        for word in requiredWords:
            if word not in oneSetOfStopWords and word not in string.punctuation:
                totalWords.append(word)

    wordfreqdist = nltk.FreqDist(totalWords)
    mostcommon = wordfreqdist.most_common(50)
    print(mostcommon)
    new_text = res['cleaned_resume'].values

    from model import requiredText
    word_vectorizer.fit(requiredText)
    new_WordFeatures = word_vectorizer.transform(new_text)
    input_val = new_WordFeatures
    print(input_val.shape)


    from model import clf
    test_pred = clf.predict(input_val)
    print(test_pred)

    # def transform(text_file_contents):
    #     return text_file_contents.replace(",", ",")
    # final = [np.array(int_features)]
    # print(int_features)
    # print(final)
    # prediction = model.predict_proba(final)
    # output = '{0:.{1}f}'.format(prediction[0][1], 2)
    # int_features=[int(x) for x in request.form.values()]


    # data = request.form.values()
    # print(data)
    # print("gnnnnnbvhmjhgfdcfghjknmjbvc", file=sys.stderr)
    # res = pd.read_csv('test.csv', encoding='utf-8')


    if (1):
        return render_template('index.html',pred='The domain of your resume is '+ test_pred)
    else:
        return render_template('index.html',pred='Your Forest is safe.\n Probability of fire occuring is {}',bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)


