import MySQLdb

from ..database import database

connection = database.DatabaseConnection()
db, cursor = connection.db, connection.cursor


def get_logs_by_date(date_of_log):
    try:
        cursor.execute(
            u"SELECT log_id, line_message, is_summary FROM GNUeIRCLogs "
            u"WHERE date_of_log=%s", (date_of_log,)
        )
        return cursor.fetchall()
    except MySQLdb.Error as error:
        print(error)
        db.rollback()


def get_log_by_id(log_id):
    try:
        cursor.execute(
            u"SELECT log_id, line_message, is_summary, date_of_log FROM GNUeIRCLogs "
            u"WHERE log_id=%s", (log_id,)
        )
        return cursor.fetchone()
    except MySQLdb.Error as error:
        print(error)
        db.rollback()
    except Exception as error:
        print error

