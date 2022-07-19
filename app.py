import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE




app= Flask(__name__)

global data, x_train, x_test, y_train, y_test

df= pd.read_csv(r'data.tsv', sep='\t')

num_var = df.select_dtypes(exclude='object')
num_var.fillna(num_var.median(),inplace = True)


cat_var = df.select_dtypes(include='object')
cat_var = cat_var.apply(lambda x: x.fillna(x.value_counts().index[0]))


le = LabelEncoder()
cat_var1 = cat_var.apply(le.fit_transform)

data = pd.concat([num_var,cat_var1],axis = 1)

X = data.drop(['diseaseType','NofSnps','EI'],axis = 1)
y = data.diseaseType


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)



x_train,x_test,y_train,y_test = train_test_split(X_res,y_res,test_size = 0.3,random_state = 23)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/training', methods= ['GET','POST'])
def training():
    x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state= 23)

    if request.method== 'POST':
        model= int(request.form['algo'])
        if model== 1:
            cfr = RandomForestClassifier()
            model = cfr.fit(x_train[:10000], y_train[:10000])
            pred = model.predict(x_test[:10000])
            rfcr= accuracy_score(y_test[:10000], pred)
            msg= "Your Accuracy is: "+ str(rfcr)



        elif model== 2:
            xgc = XGBClassifier()
            model1 = xgc.fit(x_train[:10000], y_train[:10000])
            pred = model1.predict(x_test[:10000])
            xgcr = accuracy_score(y_test[:10000], pred)
            msg = "Your Accuracy is: " + str(xgcr)


        elif model== 3:
            lgb1 = lgb.LGBMClassifier()
            model2 = lgb1.fit(x_train[:10000], y_train[:10000])
            pred2 = model2.predict(x_test[:10000])
            lgcr = accuracy_score(y_test[:10000], pred2)
            msg = "Your Accuracy is: " + str(lgcr)


        elif model== 4:
            model3 = KNeighborsClassifier()
            model3.fit(x_train[:10000], y_train[:10000])
            pred3 = model3.predict(x_test[:10000])
            accuracy_score(y_test[:10000], pred3)
            kncr = accuracy_score(y_test[:10000], pred3)
            msg = "Your Accuracy is: " + str(kncr)


        elif model== 5:
            model4 = SVC()
            model4.fit(x_train[:10000], y_train[:10000])
            pred4 = model4.predict(x_test[:10000])
            svcr = accuracy_score(y_test[:10000], pred4)
            msg = "Your Accuracy is: " + str(svcr)


        return render_template('training.html', msg = msg)
    return render_template('training.html')

@app.route('/prediction', methods= ['GET', 'POST'])
def prediction():
    if request.method== "POST":
        geneId= request.form['geneId']
        print(geneId)
        DSI= request.form['DSI']
        print(DSI)
        DPI= request.form['DPI']
        print(DPI)
        score= request.form['score']
        print(score)
        YearInitial= request.form['YearInitial']
        print(YearInitial)
        YearFinal= request.form['YearFinal']
        print(YearFinal)
        NofPmids= request.form['NofPmids']
        print(NofPmids)
        geneSymbol= request.form['geneSymbol']
        print(geneSymbol)
        diseaseId= request.form['diseaseId']
        print(diseaseId)
        diseaseName= request.form['diseaseName']
        print(diseaseName)
        diseaseClass= request.form['diseaseClass']
        print(diseaseClass)
        diseaseSemanticType= request.form['diseaseSemanticType']
        print(diseaseSemanticType)
        source= request.form['source']
        print(source)

        di= {'geneId' : [geneId], 'DSI' : [DSI], 'DPI' : [DPI], 'score' : [score], 'YearInitial' : [YearInitial],
             'YearFinal' : [YearFinal], 'NofPmids' : [NofPmids], 'geneSymbol' : [geneSymbol], 'diseaseId' : [diseaseId],
             'diseaseName' : [diseaseName], 'diseaseClass' : [diseaseClass],
             'diseaseSemanticType' : [diseaseSemanticType], 'source' : [source]}

        test= pd.DataFrame.from_dict(di)
        print(test)
        x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state= 23)
        cfr = RandomForestClassifier()
        model = cfr.fit(x_train[:10000], y_train[:10000])
        output = model.predict(test)
        print(output)

        if output[0] == 0:
            val = '<b><span style = color:black;>The Patient Has  <span style = color:red;>Disease </span></span></b>'

        elif output[0] == 1:
            val = '<b><span style = color:black;>The Patient Has  <span style = color:red;>Group </span></span></b>'

        elif output[0] ==2:
            val = '<b><span style = color:black;>The Patient Has  <span style = color:red;>Phenotype </span></span></b>'

        return render_template('prediction.html', msg=val)
    return render_template('prediction.html')



if __name__ == '__main__':
    app.run(debug=True)










