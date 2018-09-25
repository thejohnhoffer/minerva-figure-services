import cv2
import json
import base64
import logging
import numpy as np

from functools import wraps
from datetime import date, datetime
from typing import Any, Callable, Dict, Union, List

from minerva_scripts.crop import do_crop
from minerva_scripts.omeroapi import OmeroApi
from minerva_scripts.minervaapi import MinervaApi


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _event_path_param(event, key):
    '''
    return {
        'token': 'eyJraWQiOiJYT0E0b01xV1RsMzFBbGRMQUh3UXNzREoyWEg5ZnFlU015MVJaVXdSb2dvPSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI2Mjk2MmYzYy03OTI0LTRlODctYThmNS02NjY4OTEyMTlhZjUiLCJhdWQiOiI2Y3RzbmpqZ2xtdG5hMnE1Zmd0cmp1ZzQ3ayIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJ0b2tlbl91c2UiOiJpZCIsImF1dGhfdGltZSI6MTUzNzg5Nzk1OSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLnVzLWVhc3QtMS5hbWF6b25hd3MuY29tXC91cy1lYXN0LTFfWXVURjlTVDRKIiwibmFtZSI6Ik1pbmVydmEgR3Vlc3QiLCJjb2duaXRvOnVzZXJuYW1lIjoiNjI5NjJmM2MtNzkyNC00ZTg3LWE4ZjUtNjY2ODkxMjE5YWY1IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiZ3Vlc3QiLCJleHAiOjE1Mzc5MDE1NTksImlhdCI6MTUzNzg5Nzk1OSwiZW1haWwiOiJqb2huQGhvZmYuaW4ifQ.mgtgoWVyIAxx1foU60k71heG8f5Ocp8K1eLLc1U1ldGlcmn7LikNl1geFW3hsy9bU2t0VKVQsnFl2CxDc82qrWviQZLTVsTL9kkpTOc_n5I7MuBD-ByXCyqymgAMnc3xWWnBcwDKVuy8vQvutaUWbEwgl2StPItnJ7il4qLViGaSNeT5b3mkzmg84Zpl4hbB_DNx5wVxjdt1jjcMT4Q6RNl-vB0DXVsRBeNFDAW341YwSj1tZ_ywWpK1z1xP3mAJNJLce-tqvtfEgyqVdVfwFC0j_TdXFpRa-P5NYh89YWKz6aoLiQB70-YfW-7EwrW7thJMjXE5WFODvYS3kUadog',
        'uuid': 'afd6f4bd-67de-4df2-b518-0e9b05a49012',
        'z': '0',
        't': '0'
    }[key]
    '''
    return event['pathParameters'][key]


def _event_query_params(event):
    '''
    query_params = event['queryStringParameters']
    return {
        'c': '1|13746:22528$FFFFFF',
        'maps': '[{"reverse":{"enabled":false}}]',
        'm': 'c'
    }
    '''
    return event['queryStringParameters']


def json_custom(obj: Any) -> str:
    '''JSON serializer for extra types.
    '''

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type {} not serializable".format(type(obj)))


def make_response(code: int, body: Union[Dict, List]) -> Dict[str, Any]:
    '''Build a response.
        Args:
            code: HTTP response code.
            body: Python dictionary or list to jsonify.
        Returns:
            Response object compatible with AWS Lambda Proxy Integration
    '''

    return {
        'statusCode': code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true'
        },
        'body': json.dumps(body, default=json_custom)
    }


def make_binary_response(code: int, body: np.ndarray) -> Dict[str, Any]:
    '''Build a binary response.
        Args:
            code: HTTP response code.
            body: Numpy array representing image.
        Returns:
            Response object compatible with AWS Lambda Proxy Integration
    '''

    encoded = base64.b64encode(body).decode('utf-8')
    response = {
        'statusCode': code,
        'headers': {
            'Content-Type': 'image/png',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true'
        },
        'body': encoded,
        'isBase64Encoded': True
    }
    return response


def _event_body(event):
    if 'body' in event and event['body'] is not None:
        return json.loads(event['body'])
    return {}


def response(code: int) -> Callable[..., Dict[str, Any]]:
    '''Decorator for turning exceptions into responses.
    KeyErrors are assumed to be missing parameters (either query or path) and
    mapped to 400.
    ValueErrors are assumed to be parameters (either query or path) that fail
    validation and mapped to 422.
    Any other Exceptions are unknown and mapped to 500.
    Args:
        code: HTTP status code.
    Returns:
        Function which returns a response object compatible with AWS Lambda
        Proxy Integration.
    '''

    def wrapper(fn):
        @wraps(fn)
        def wrapped(self, event, context):

            # Execute the function and make a response or error response
            try:
                self.body = _event_body(event)
                return make_binary_response(code, fn(self, event, context))
            except KeyError as e:
                return make_response(400, {'error': str(e)})
            except ValueError as e:
                return make_response(422, {'error': str(e)})
            except Exception as e:
                logger.exception(e)
                return make_response(500, {'error': str(e)})

        return wrapped
    return wrapper


class Handler:

    token = None
    uuid = None
    z = 0
    t = 0
    bucket = 'minerva-test-cf-common-tilebucket-1su418jflefem'
    domain = 'lze4t3ladb.execute-api.us-east-1.amazonaws.com/dev'

    def load_tile(self, c, l, y, x):
        ''' Minerva loads a single tile '''

        token = self.token
        uuid = self.uuid
        keywords = {
            't': 0,
            'z': 0,
            'l': l,
            'x': x,
            'y': y
        }
        return MinervaApi.image(uuid, token, c, None, **keywords)

    @response(200)
    def render(self, event, context):
        '''Render the specified tile with the given settings'''

        # Path and Query parameters
        query_dict = _event_query_params(event)
        self.token = _event_path_param(event, 'token')
        self.uuid = _event_path_param(event, 'uuid')
        self.z = _event_path_param(event, 'z')
        self.t = _event_path_param(event, 't')

        keys = OmeroApi.scaled_region([self.uuid, self.z, self.t],
                                      query_dict, self.token,
                                      self.bucket, self.domain)

        # Make array of channel parameters
        inputs = zip(keys['chan'], keys['c'], keys['r'])
        channels = map(MinervaApi.format_input, inputs)

        # Region with margins
        outer_origin = keys['origin']
        outer_shape = keys['shape']
        outer_end = np.array(outer_origin) + outer_shape
        out_h, out_w = outer_shape.astype(np.int64)
        out = np.ones((out_h, out_w, 3)) * 0.5

        # Actual image content
        image_shape = keys['image_shape']
        request_origin = np.maximum(outer_origin, 0)
        request_end = np.minimum(outer_end, image_shape)
        request_shape = request_end - request_origin

        # Minerva does the cropping
        image = do_crop(self.load_tile, channels, keys['tile_shape'],
                        request_origin, request_shape, keys['levels'],
                        keys['max_size'])

        # Position cropped region within margins
        y_0, x_0 = (request_origin - outer_origin).astype(np.int64)
        y_1, x_1 = [y_0, x_0] + request_shape
        out[y_0:y_1, x_0:x_1] = image

        # Return encoded png
        output = (255 * out).astype(np.uint8)[:, :, ::-1]
        png_output = cv2.imencode('.png', output)[1]
        print('encoded length is ', len(png_output))
        return png_output


handler = Handler()
render = handler.render
