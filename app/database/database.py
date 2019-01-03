import MySQLdb
import MySQLdb.cursors as cursors


class DatabaseConnection(object):

    def __init__(self):
        self.db = MySQLdb.connect(
            host="127.0.0.1",
            user="root",
            passwd="bubo1234",
            db="gnue_irc",
            use_unicode=True,
            charset="utf8",
            cursorclass=cursors.DictCursor
        )
        self.cursor = self.db.cursor()
        self.cursor.execute('SET NAMES utf8mb4')
        self.cursor.execute('SET CHARACTER SET utf8mb4')
        self.cursor.execute('SET character_set_connection=utf8mb4')
