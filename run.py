from flask import Flask, jsonify, request, render_template

from .app.services.logs import get_logs_by_date, get_log_by_id, update_log_by_id

app = Flask(__name__, template_folder='app/templates')


@app.route("/")
def hello_world():
    return render_template('index.html', my_string="Some string!", my_list=[0,1,2,3,4,5])


@app.route("/logs/date/<date>")
def get_all_logs_for_given_date(date):
    logs = get_logs_by_date(date)
    logs = {"logs": logs}
    return jsonify(logs)


@app.route("/logs/<log_id>", methods=["GET", "PUT"])
def get_log(log_id):
    if request.method == "GET":
        log_info = get_log_by_id(log_id)
        return jsonify(log_info)
    elif request.method == "PUT":
        is_summary = request.form.get('is_summary')
        if not is_summary:
            return jsonify({"error": "is_summary field is required"})
        update_log_by_id(log_id, is_summary)
        return jsonify({"success": "successfully updated is_summary field"})

