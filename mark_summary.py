import MySQLdb

from database import DatabaseConnection


def get_quotes_for_date(quote_date):
    """
    Get all quotes for a particular IRC chat date from the database
    :param quote_date: A string representing the date in the format YYYY-MM-DD
    :return: A hash set containing all quotes for a particular date
    """
    quotes_set = set()
    connection = DatabaseConnection()
    db, cursor = connection.db, connection.cursor

    try:
        cursor.execute(
            u"SELECT quote FROM GNUeSummaryParaQuotes "
            u"WHERE quote_date=%s", (quote_date,)
        )
        para_quotes = cursor.fetchall()

        for quote in para_quotes:
            quotes = quote[0].split('-')
            for q in quotes:
                q = q.rstrip()
                q = q.replace('\n', '')
                quotes_set.add(q)

        print(quotes_set)

    except MySQLdb.Error as error:
        print(error)
        db.rollback()


# get_quotes_for_date("2002-04-10")


