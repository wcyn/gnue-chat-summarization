import io
import MySQLdb

from bs4 import BeautifulSoup
from app.database import database
from os import listdir
from os.path import isfile, join


connection = database.DatabaseConnection()
db, cursor = connection.db, connection.cursor


def create_logs_ids_map(date_of_log):
    log_ids_map = {}
    try:
        cursor.execute(
            u"SELECT log_id, line_message FROM GNUeIRCLogs "
            u"WHERE date_of_log=%s", (date_of_log,)
        )
        for log_id, line_message in cursor:
            line_message = line_message.strip()
            line_message = line_message.replace("\n", "")
            line_message = " ".join(line_message.split())
            log_ids_map[line_message] = log_id

    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        db.rollback()
    return log_ids_map


def add_log_ids_to_line_messages():
    path = 'summarizedLogs'
    files = [f for f in listdir(path) if isfile(join(path, f))][:1]

    for filename in files:
        log_date = filename.split(".")[0]
        log_ids_map = create_logs_ids_map(log_date)
        print("reading %s: %s" % (filename, log_date))
        path_to_file = join(path, filename)
        html = io.open(path_to_file, "r", encoding="utf8")
        soup = BeautifulSoup(html, features="html.parser")


        for row in soup.find_all('tr'):
            line_message = row.find("td", **{"class": "text"})
            if line_message:
                line_text = line_message.find(text=True, recursive=False)
                line_text = line_text.strip()
                line_text = line_text.replace("\n", "")
                line_text = " ".join(line_text.split())
                # print line_text
                if line_text in log_ids_map:
                    # print line_text
                    tooltip_text = soup.new_tag("span", **{"class": "tooltip-text"})
                    checkbox_label = soup.new_tag("label")
                    # checkbox_label.string = " Quoted?"
                    quoted_checkbox = soup.new_tag("input", **{
                        "type": "checkbox",
                        "value": str(log_ids_map[line_text])
                    })
                    checkbox_label.append(quoted_checkbox)
                    tooltip_text.string = str(log_ids_map[line_text])
                    line_message.append(tooltip_text)
                    line_message.append(checkbox_label)
                    line_message["log_id"] = log_ids_map[line_text]
                    line_message["class"] = "text tooltip"
                    # print line_message
        with open(path_to_file, "w") as f:
            f.write(str(soup))

add_log_ids_to_line_messages()
