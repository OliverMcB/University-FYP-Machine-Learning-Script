from flask import request
from flask_restful import Resource
import json

import pandas as pd
import BreastCancerDeepLearning.neuralnetworkmodel as nnm


class Predict(Resource):

    def get(self):
        data = request.form["data"]

        data = pd.read_json(data)

        result = json.dumps(nnm.predict(data).tolist())

        return result, 200
