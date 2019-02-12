import MySQLdb
import MySQLdb.cursors as cursors
import os

class DatabaseConnection:

    def __init__(self):
        self.db = MySQLdb.connect(
            host=os.environ.get("DB_HOST"),
            user=os.environ.get("DB_USER"),
            passwd=os.environ.get("DB_PASS"),
            db=os.environ.get("DB_NAME"),
            use_unicode=True,
            charset="utf8",
            cursorclass=cursors.DictCursor
        )
        self.cursor = self.db.cursor()
        self.cursor.execute('SET NAMES utf8mb4')
        self.cursor.execute('SET CHARACTER SET utf8mb4')
        self.cursor.execute('SET character_set_connection=utf8mb4')
