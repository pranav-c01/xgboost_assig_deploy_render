# from wsgiref import simple_server
from flask import Flask, request, app
from flask import Response
from flask_cors import CORS
from logistic_deploy import predObj

# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle


application = Flask(__name__)
# CORS(application)
# application.config['DEBUG'] = True

@application.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


# class ClientApi:~

#     def __init__(self):
#         self.predObj = predObj()

@application.route("/predict_api", methods=['POST'])
def predictRoute():
    try:
        if request.json['data'] is not None:
            data = request.json['data']
            print('data is:     ', data)
            pred=predObj()
            res = pred.predict_log(data)

            print('result is        ',res)
            return Response(res)
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ',e)
        return Response(e)

@application.route("/predict", methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            native_country=float(request.form['native_country'])
            hours_per_week = float(request.form['hours_per_week'])
            capital_loss = float(request.form['capital_loss'])
            capital_gain = float(request.form['capital_gain'])
            sex = float(request.form['sex'])
            educ = float(request.form['educ'])
            education_num = float(request.form['education_num'])
            workclass = float(request.form['workclass'])
            predict_dict ={"native_country": native_country,
                "hours_per_week": hours_per_week,
                "capital_loss" : capital_loss,
                "capital_gain": capital_gain,
                "sex": sex,
                "educ":educ,
                "education_num":education_num,
                "workclass":workclass
    }
            pred2=predObj()
            res = pred2.predict_log(predict_dict)
            print('result is        ',res)
            # showing the diagnosed results in a UI
            return render_template('results.html',prediction=res)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')

if __name__ == "__main__":
    # clntApp = ClientApi()
    # host = '0.0.0.0'
    # port = 5000
    application.run(debug=True)
    #httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()