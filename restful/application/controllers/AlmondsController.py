from restful.application.logic.functions import read_image, read_image_from_endpoint
import numpy as np
import cv2 as cv2
from jsonpickle import encode
from flask import Response
import codecs
import os
import json
from json import JSONEncoder

from restful.application.models.NumpyArrayEncoder import NumpyArrayEncoder


class AlmondsController(object):

    @staticmethod
    def processAlmond(request):
        # sizing of the images
        nrows = 224
        ncolumns = 224
        channels = 3  # change to 1 if you want to use grayscale image

        # check if the post request has the file part
        if 'image' not in request.files:
            return {"detail": "No image found"}, 400
        file = request.files['image']
        if file.filename == '':
            return {"detail": "Invalid file or filename missing"}, 400

        # read the image
        image = request.files['image']
        # decode image
        bytes_as_np_array = np.frombuffer(image.read(), dtype=np.uint8)
        img = cv2.imdecode(bytes_as_np_array, cv2.IMREAD_COLOR)
        img = read_image_from_endpoint(img, nrows, ncolumns)

        response = {'image_size': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                    'image_numpy': str(np.expand_dims(img, axis=0).tolist())}
        encodedNumpyData = json.dumps(response, cls=NumpyArrayEncoder)
        return Response(response=encodedNumpyData, status=200, mimetype="application/json")
