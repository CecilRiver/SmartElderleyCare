import pymysql


class DatabaseConnector:
    def __init__(self, host, port, database, user, password, charset='utf8'):
        self.connection = pymysql.connect(host=host, port=port, database=database, user=user, password=password, charset=charset)
        self.cur = self.connection.cursor()

    def insertIntoAction(self, actionTime, actionType, actionPicture):
        sql = "INSERT INTO action (actionTime, actionType, actionPicture) VALUES (%s, %s, %s)"
        values = (actionTime, actionType, actionPicture)
        self.cur.execute(sql, values)
        self.connection.commit()

    def getActionData(self):
        sql = "SELECT * FROM action"
        self.cur.execute(sql)
        data = self.cur.fetchall()
        return data

    def closeConnection(self):
        self.cur.close()
        self.connection.close()


