import logging

import flask
from flask import Flask, request, jsonify, Response, render_template
from predict import predict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Initialize the Flask application
application = Flask(__name__, template_folder="app/templates")

def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp

@application.route('/')
def index():
    return render_template('index.html')


@application.route('/classify', methods=['POST'])
def sentiment_classification():
    json_request = request.get_json()
    if not json_request:
        return Response("No json provided.", status=400)
    # text = json_request['text']
    text = json_request
    if text is None:
        return Response("No text provided.", status=400)
    else:
        label = predict(text)
        return label
        # return flask.jsonify({"text": text, "label": label})


if __name__ == '__main__':
    application.run(debug=True, use_reloader=True)