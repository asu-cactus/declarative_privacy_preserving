from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import pandas as pd
import random
from api import run


# def run(
#     learning_rate,
#     epochs,
#     l2_norm_clip,
#     noise_multiplier,
#     batch_size,
#     delta,
#     units
# ) -> tuple[float, float]:
#     return tuple[random.randint(0, 100), random.randint(0, 100)]


app = Flask(__name__)
api = Api(app)
CORS(app)


class Users(Resource):
    def get(self):
        data = "INFORMATION FROM HONG"
        return {'data': data}, 200  # return data and 200 OK code


class Locations(Resource):
    def get(self):
        data = pd.read_csv('locations.csv')  # read CSV
        data = data.to_dict()  # convert dataframe to dictionary
        return {'data': data}, 200  # return data and 200 OK code

    def post(self):
        schema_parameters = request.get_json()
        print(f'Parameters from frontend:\n{schema_parameters}')
        # learning_rate = schema_parameters['learning_rate']
        # epochs = schema_parameters['epochs']
        # l2_norm_clip = schema_parameters['l2_norm_clip']
        # noise_multiplier = schema_parameters['noise_multiplier']
        # batch_size = schema_parameters['batch_size']
        # delta = schema_parameters['delta']
        # unit = schema_parameters['units']
        # result = run(learning_rate, epochs, l2_norm_clip,
        #              noise_multiplier, batch_size, delta, unit)
        results = run(**schema_parameters)
        print(f'Results:\n{results}')
        return {'data': results}, 200
        # return {'data': {'location': 'Food Court, Phoenix Sky Harbor Airport', 'privacy_budget': '5342',
        #                  'accuracy': '27.8%'}}, 200


api.add_resource(Users, '/users')  # '/users' is our entry point for Users
# and '/locations' is our entry point for Locations
api.add_resource(Locations, '/locations')

if __name__ == '__main__':
    app.run()  # run our Flask app


# curl -d '{"learning_rate": 0.02,"epochs": 70,"l2_norm_clip": 1,"noise_multiplier": 0.3,"batch_size": 200,"delta": 1e-5,"units": 500}' -H 'Content-Type: application/json' http://127.0.0.1:5000/locations
