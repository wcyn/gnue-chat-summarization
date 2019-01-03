from flask import Flask, jsonify

from .app.services.logs import get_logs_by_date, get_log_by_id

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Welcome to the Conversational Data Preprocessor homepage"


@app.route("/logs/date/<date>")
def get_all_logs_for_given_date(date):
    logs = get_logs_by_date(date)
    logs = {"logs": logs}
    return jsonify(logs)


@app.route("/logs/<log_id>/")
def get_log(log_id):
    log_info = get_log_by_id(log_id)
    return jsonify(log_info)
