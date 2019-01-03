import re
import codecs
import MySQLdb

from app.database.database import DatabaseConnection
from os import listdir
from os.path import isfile, join

datasource_id = 48038
# Open database connection
connection = DatabaseConnection()
db, cursor = connection.db, connection.cursor

path = 'chatLogs'
files = [f for f in listdir(path) if not f.startswith('.') and isfile(
    join(path, f))]
for filename in files:
    print('reading %s' % (filename))
    log = codecs.open(
        join(path, filename), 'r', encoding='utf-8', errors='ignore'
        )
    line_count = 0
    username = ''
    line_message = ''
    line_type = ''
    for line in log:
        line_count += 1
        # case 1: user message
        patternMessage = re.compile(ur'^<(.+?)>\s(.+?)$', re.UNICODE)
        # case 2: newer system message with *** at the front
        patternSystem = re.compile(ur'^\*\*\*\s(.+?)$', re.UNICODE)

        if patternMessage.search(line):
            username = patternMessage.search(line).group(1)
            line_message = patternMessage.search(line).group(2)
            line_type = 'message'
            try:
                cursor.execute(
                    u"INSERT INTO GNUeIRCLogs(line_count, line_type, send_user,\
                    line_message, datasource_id, date_of_log) \
                    VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        line_count, line_type, username, line_message,
                        datasource_id, filename
                    )
                )
                db.commit()
                username = ''
                line_message = ''
            except MySQLdb.Error as error:
                print(error)
                db.rollback()

        # elif patternSystem.search(line):
        #     line_message = patternSystem.search(line).group(1)
        #     line_type = 'system'
        # else:
        #     line_type = 'system'
        #     line_message = line

        # print 'line_count:', line_count, ': ', line_type

    log.close()
# disconnect from server
db.close()
