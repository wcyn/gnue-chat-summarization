from flask import Flask, jsonify, request, render_template, url_for, redirect

from .services.logs import (
    get_logs_by_date,
    get_log_by_id,
    get_summary_by_date,
    get_username_colors,
    update_log_by_id,
    update_log_message_summaries
)

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/")
def home_page():
    return render_template(
        'index.html',
        app_name="GNUe IRC Chat Logs Preprocessor",
        endpoints=[("/logs/date/<date>", ["GET"]), ("/logs/<log_id>", ["GET", "PUT"])])


@app.route("/p/logs/<date>", methods=["GET", "POST"])
def logs_page(date):

    logs = get_logs_by_date(date)
    usernames = {log.get('send_user') for log in logs}
    username_colors = get_username_colors(usernames)
    summary = get_summary_by_date(date)
    if request.method == "GET":
        return render_template(
            'logs.html',
            date=date,
            logs=logs,
            username_colors=username_colors,
            summary=summary
        )
    elif request.method == "POST":
        result = request.form
        summary_messages = result.getlist('chat_log')
        update_log_message_summaries(date, summary_messages)
        print summary_messages
        return redirect(url_for('logs_page', **{'date': date}))


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

