from flask import Flask
from flask import request
from flask_restful import reqparse, Api, Resource
import pickle

from model.convert_to_json import preprocess_data

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('post')

articles = ['business', 'entertainment', 'politics', 'sport', 'tech']

# load vectorizer
vectorizer = pickle.load(open("../model/vectorizer.pickle", 'rb'))

# load model
with open('../model/text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)


class PredictTheme(Resource):
    def post(self):
        data = request.json
        data = preprocess_data(data['post'])

        data = vectorizer.transform([data]).toarray()

        res = model.predict(data)

        return {'prediction': articles[res[0]]}


api.add_resource(PredictTheme, '/')

if __name__ == '__main__':
    app.run(debug=True)
