from flask import Flask, render_template,url_for, request, session, redirect,g,make_response,Response
import os
#from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import sys
import csv
app = Flask(__name__)
#app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///'

app.secret_key = os.urandom(24)

data = pd.read_csv("nursingfinalds.csv")


def manage_data(data, train_percent, test_percent):
    cols = data.columns

    distinct_class_val = data[cols[-1]].unique()

    data_dic = {}

    index_start = 0

    for i in distinct_class_val:
        datas = data[data[cols[-1]] == distinct_class_val[index_start]][:500].values.tolist()

        total_rows = len(datas)

        # datas = np.concatenate([datas,np.reshape(([index_start] * total_rows),(total_rows,1))],axis=1)

        train_no = round(total_rows * (train_percent / 100))
        test_no = round(total_rows * (test_percent / 100))
        data_dic.update({i: {'train': datas[:train_no], 'test': datas[train_no:]}})

        index_start += 1

    merge_train = data_dic[distinct_class_val[0]]['train']

    merge_test = data_dic[distinct_class_val[0]]['test']

    for i in range(len(distinct_class_val) - 1):
        merge_train = np.concatenate((merge_train, data_dic[distinct_class_val[i + 1]]['train']))

        merge_test = np.concatenate((merge_test, data_dic[distinct_class_val[i + 1]]['test']))

    return merge_train, merge_test



merge_train, merge_test = manage_data(data, 80, 20)
def euclidean_distance(row1, row2):
    distance = 0.0

    for i in range(len(row1) - 1):
        distance += (float(row1[i]) - float(row2[i])) ** 2

    return np.sqrt(distance)


def manhattan_distance(row1, row2):
    distance = 0.0

    for i in range(len(row1) - 1):
        distance += abs(float(row1[i]) - float(row2[i]))

    return distance


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors, distance_formula):
    distances = list()

    for train_row in train:
        if distance_formula == 'euclidean':
            dist = euclidean_distance(test_row, train_row)
        elif distance_formula == 'manhattan':
            dist = manhattan_distance(test_row, train_row)

        distances.append((train_row, dist))

    distances.sort(key=lambda tup: tup[1])

    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors


# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors, distance_formula):
    neighbors = get_neighbors(train, test_row, num_neighbors, distance_formula)

    output_values = [row[-1] for row in neighbors]

    prediction = max(set(output_values), key=output_values.count)

    return prediction


@app.route('/')

def about():

    return render_template('index.html')

@app.route('/login', methods=["POST","GET"])

def login():

    if request.method =='POST':
        session.pop('user',None)
        if request.form['email'] == "nursing@info.com" and request.form['password'] == "123456":
            session['logins'] = request.form['email']
            return redirect(url_for("home"))
        else:
            return render_template('login.html',wrongs=True)

    return render_template('login.html')

@app.route('/home')

def home():
    if g.logins:
        filename = "tested.csv"
        data =[]
        with open(filename) as infile:
            csvfile = csv.reader(infile)
            for row in csvfile:
                data.append(row)


        return render_template('home.html',useris=session['logins'],data =data)
    return redirect(url_for('login'))

@app.before_request

def before_request():
    g.logins = None

    if 'logins' in session:
        g.logins = session['logins']

@app.route('/dropsession')
def dropsession():
    session.pop('logins',None)

    return render_template('login.html')

@app.route("/dlcsv")
def getPlotCSV():
    csv = ''

    with open("tested.csv") as fp:
         csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=tested.csv"})


@app.route('/covid-testing',methods=["POST","GET"])

def covid_testing():

    if request.method == 'POST':
        fullname = request.form['fname']
        email = request.form['email']
        address = request.form['Address']
        contactno = request.form['Contactnumber']
        fever = request.form['Fever']
        tiredness = request.form['Tiredness']
        drycough = request.form['Dry-Cough']
        difficultb = request.form['Difficulty-in-Breathing']
        sorethroat = request.form['Sore-Throat']
        pains = request.form['Pains']
        nasalconges = request.form['Nasal-Congestion']
        diarrhea = request.form['Diarrhea']


        test_set = [fever,tiredness,drycough,difficultb,sorethroat,pains,nasalconges,diarrhea]

        predicted =  predict_classification(merge_train, test_set, 1, 'euclidean')

        filename = "tested.csv"

        toadd= [fullname,email,address,contactno,predicted]

        with open(filename, "r") as infile:
            reader = list(csv.reader(infile))
            reader.insert(1, toadd)

        with open(filename, "w", newline='') as outfile:
            writer = csv.writer(outfile)
            for line in reader:
                writer.writerow(line)

        return render_template('covid-testing.html', user=predicted)

    else:
        return render_template('covid-testing.html')


if __name__ == "__main__":
    app.run(debug=True)