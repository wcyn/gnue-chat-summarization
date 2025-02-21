import time

from datetime import date, timedelta
from flask import Flask, jsonify, request, render_template, url_for, redirect
from .services.logs import (
    get_logs_by_date,
    get_log_by_id,
    get_summary_and_quotes_by_date,
    get_username_colors,
    generate_random_color,
    format_date_data_types,
    update_log_by_id,
    update_log_message_summaries,
    get_conversation_statistics_by_date,
    get_date_data_types,
    get_prediction_clusters
)

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/")
def home_page():
    date_data_types = get_date_data_types()
    return render_template(
        'index.html',
        app_name="GNUe IRC Chat Logs Preprocessor",
        date_data_types=format_date_data_types(date_data_types),
        endpoints=[("/logs/date/<date>", ["GET"]), ("/logs/<log_id>", ["GET", "PUT"])])


@app.route("/p/logs/<logs_date>", methods=["GET", "POST"])
def logs_page(logs_date):

    logs = get_logs_by_date(logs_date)
    if request.method == "GET":
        conversation_statistics = get_conversation_statistics_by_date(logs_date)
        username_colors = {log.get('send_user'): generate_random_color() for log in logs}
        summary_and_quotes = get_summary_and_quotes_by_date(logs_date)
        current_date = time.strptime(logs_date, '%Y-%m-%d')
        previous_date = (date(
            current_date.tm_year, current_date.tm_mon, current_date.tm_mday) - timedelta(1)).strftime('%Y-%m-%d')
        next_date = (date(
            current_date.tm_year, current_date.tm_mon, current_date.tm_mday) + timedelta(1)).strftime('%Y-%m-%d')

        return render_template(
            'logs.html',
            logs_date=logs_date,
            logs=logs,
            prediction_clusters=get_prediction_clusters(logs),
            conversation_statistics=conversation_statistics,
            username_colors=username_colors,
            summary_and_quotes=summary_and_quotes,
            previous_date=previous_date,
            next_date=next_date
        )
    elif request.method == "POST":
        result = request.form
        summary_log_ids = set(result.getlist('chat_log'))
        update_log_message_summaries(logs_date, logs, summary_log_ids)
        print(summary_log_ids)
        return redirect(url_for('logs_page', **{'logs_date': logs_date}))


@app.route("/logs/date/<date>")
def get_all_logs_for_given_date(logs_date):
    logs = get_logs_by_date(logs_date)
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

