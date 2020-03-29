from flask import request
from flask_restful import Resource

import BreastCancerDeepLearning.neuralnetworkmodel as nnm


class Predict(Resource):

    def get(self):

        data = request.form["data"]

        nnm.initialise()

        result = nnm.predict(data)

        return result, 200
