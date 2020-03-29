from flask import Flask
from flask_restful import Api

from BreastCancerDeepLearning.Predict import Predict

app = Flask(__name__)
api = Api(app)

api.add_resource(Predict, "/predict")

if __name__ == "__main__":
    app.run()