import logging
from flask import Flask, request, Response, Request
from flask_restful import Api
from restful.application.config.config import ProductionConfig, DevelopmentConfig
import os
from restful.application.controllers import AlmondsController
import numpy as np


def create_app(config_name):
    flask_app = Flask(__name__)
    flask_app.config.from_object(ProductionConfig if (config_name == 'production') else DevelopmentConfig)
    api = Api(flask_app)

    # ***** Error Handling and Logging ***** #
    logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    # ******** End-points ********#
    @api.app.route('/v1/almonds/processor', methods=['POST'])
    def processAlmondImage():
        return AlmondsController.AlmondsController.processAlmond(request)

    @flask_app.after_request
    def after_request(response):
        response.headers.set('Access-Control-Expose-Headers', 'Link')
        response.headers.set('Access-Control-Allow-Origin', '*')
        response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.set('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
        return response

    return flask_app


app = create_app(os.getenv("FLASK_ENV"))
if __name__ == '__main__':
    app.run(host="0.0.0.0", port="3001")
