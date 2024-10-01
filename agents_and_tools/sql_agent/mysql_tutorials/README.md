## SQL with python

- connection.cursor() -> This is the function that creates a cursor. A cursor is a object which can be used to execute MySQL query.
- connection.commit() -> Applies all the changes to the database that sql query indicates.
- connection.close() -> closes the connection to the mysql server.
- connection.is_connected() -> Checks if the connection is still continues or not.
- cursor.execute(<sql_query>) -> This is the function that executes the sql query.
- cursor.close() -> Closes the cursor. After executing this function cursor will not able to run sql queries.