from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Welcome to the Conversational Data Preprocessor homepage"


@app.route("/logs/date/<date>")
def get_all_logs_for_date(date):
    return "Getting all logs for date {}".format(date)


@app.route("/logs/<log_id>/")
def get_log(log_id):
    return "Log {}".format(log_id)
