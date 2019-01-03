import MySQLdb

from app.database.database import DatabaseConnection

connection = DatabaseConnection()
db, cursor = connection.db, connection.cursor


def get_quotes_for_date(quote_date):
    """
    Get all quotes for a particular IRC chat date from the database
    :param quote_date: A string representing the date in the format YYYY-MM-DD
    :return: A hash set containing all quotes for a particular date
    """
    quotes_set = set()

    try:
        cursor.execute(
            u"SELECT quote FROM GNUeSummaryParaQuotes "
            u"WHERE quote_date=%s", (quote_date,)
        )
        para_quotes = cursor.fetchall()

        for (quote,) in para_quotes:
            quotes = quote.split('-')
            for q in quotes:
                q = q.strip()
                q = q.replace('\n', '')
                q = ' '.join(q.split())
                quotes_set.add(q)

        # print(quotes_set)

    except MySQLdb.Error as error:
        print(error)
        db.rollback()

    return quotes_set


# get_quotes_for_date("2002-04-10")

def get_log_ids_of_quoted_logs(date_of_log):
    quotes_set = get_quotes_for_date(date_of_log)
    quoted_log_ids = []

    try:
        cursor.execute(
            u"SELECT log_id, line_message FROM GNUeIRCLogs "
            u"WHERE date_of_log=%s", (date_of_log,)
        )
        for log_id, line_message in cursor.fetchall():
            line_message = line_message.strip()
            line_message = line_message.replace('\n', '')
            line_message = ' '.join(line_message.split())
            if line_message in quotes_set:
                quoted_log_ids.append(int(log_id))
                # print log_id, line_message

    except MySQLdb.Error as error:
        print(error)
        db.rollback()
    return quoted_log_ids


def mark_quoted_logs_as_summary_per_date(date_of_log):
    log_ids = get_log_ids_of_quoted_logs(date_of_log)
    format_ids = ','.join(['%s'] * len(log_ids))
    log_ids.append(date_of_log)
    print date_of_log
    if len(log_ids) > 1:
        try:
            cursor.execute(
                u"UPDATE GNUeIRCLogs SET is_summary=1 "
                u"WHERE log_id IN ({}) "
                u"AND date_of_log=%s".format(format_ids), tuple(log_ids)
            )

        except MySQLdb.Error as error:
            print(error)
    else:
        print "No quotes found for: ", date_of_log


def mark_all_quoted_logs_as_summary():
    try:
        cursor.execute(
            u"SELECT date_of_log FROM GNUeIRCLogs "
            u"GROUP BY date_of_log ORDER BY date_of_log"
        )
        # print list(cursor)
        for (date_of_log,) in cursor.fetchall():
            if "2001-10-23" <= date_of_log <= "2006-09-21":
                # print date_of_log
                print 'Processing Log: ', date_of_log
                mark_quoted_logs_as_summary_per_date(date_of_log)

    except MySQLdb.Error as error:
        print(error)
        db.rollback()


mark_all_quoted_logs_as_summary()
# mark_quoted_logs_as_summary_per_date("2001-10-23")

# disconnect from server
db.close()
