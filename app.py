from flask import Flask
from flask_restful import Api
from flask_cors import CORS, cross_origin

from BreastCancerDeepLearning.Predict import Predict

app = Flask(__name__)
CORS(app)

api = Api(app)

api.add_resource(Predict, "/predict")

if __name__ == "__main__":
    app.run()
