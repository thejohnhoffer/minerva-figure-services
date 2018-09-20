import cv2
import json
import base64
import logging
import numpy as np

from functools import wraps
from datetime import date, datetime
from typing import Any, Callable, Dict, Union, List


logger = logging.getLogger()
logger.setLevel(logging.INFO)


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

    return {
        'statusCode': code,
        'headers': {
            'Content-Type': 'image/png',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true'
        },
        'body': base64.b64encode(body).decode('utf-8'),
        'isBase64Encoded': True
    }


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

            # Execute ta function and make a response or error response
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

    @response(200)
    def render(self, event, context):
        '''Render the specified tile with the given settings'''

        # CV2 requires 0 - 255 values
        composite = np.zeros((512, 512, 3), dtype=np.uint8)
        composite *= 255

        return cv2.imencode('.png', composite)[1]


handler = Handler()
render = handler.render
