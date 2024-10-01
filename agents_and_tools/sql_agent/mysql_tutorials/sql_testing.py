import mysql.connector
from datetime import timedelta, datetime


class MySQLDatabase:
    def __init__(self, host:str, user:str, password:str, database_name:str, table_name:str):
        # initializing the arguments
        self.__host = host
        self.__user = user
        self.__password = password
        self.database_name = database_name
        self.table_name = table_name

        # Connect to MySQL server
        try:
            conn = mysql.connector.connect(host=self.__host,
                                           user=self.__user,
                                           password=self.__password)
            cursor = conn.cursor()
            print(f"Connection established successfully\n")

            # getting all the database names
            cursor.execute('''Show databases''')
            self.list_database = []
            for item in cursor.fetchall():
                self.list_database.append(item[0])
        except Exception as e:
            print(f"Error occurred while tried to established connection.\nError message:\n{e}\n")
            exit(1)

        # create database and table if not exists
        if self.database_name not in self.list_database:
            cursor.execute(f"CREATE DATABASE {self.database_name}")   # creating a database while initializing
            conn.database = self.database_name
            cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        phone_number VARCHAR(15) PRIMARY KEY,
                        person_name VARCHAR(100) NOT NULL,
                        age INT,
                        appointment_date DATE NOT NULL,
                        appointment_time TIME NOT NULL,
                        appointment_end_time TIME
                    )''')
            print(f"Database and table is created with the name --{self.database_name}-- and --{self.table_name}--\n")
            if conn.is_connected():
                conn.commit()
                cursor.close()
        else:
            print(f"Database with the name --{self.database_name}-- already exists. Initiated database.\n")

    
    # private function for establishing connection with the database
    def __get_connection_cursor(self):
        try:
            conn = mysql.connector.connect(host=self.__host,
                                        user=self.__user,
                                        password=self.__password,
                                        database=self.database_name)
            return conn, conn.cursor()
        except Exception as e:
            print(f"Error occurred while tried to established connection.\nError message:\n{e}\n")
            return None, None


    # private function to format result
    def __format_search_result(self, result):
        if result is None:
            return None
        phone_number, person_name, age, appointment_date, appointment_time, appointment_end_time = result
        formatted_date = appointment_date.strftime("%d-%m-%Y")
        formatted_appointment_time = str(timedelta(seconds=appointment_time.seconds))[:-3]
        formatted_appointment_end_time = str(timedelta(seconds=appointment_end_time.seconds))[:-3]
        return {
            "phone_number": phone_number,
            "person_name": person_name,
            "age": age,
            "appointment_date": formatted_date,
            "appointment_time": formatted_appointment_time,
            "appointment_end_time": formatted_appointment_end_time
        }
    

    # function for deleting table
    def delete_table(self):
        try:
            conn, cursor = self.__get_connection_cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")   # sql query for deleting table
            conn.commit()

            if conn.is_connected():
                cursor.close()
                conn.close()
            print(f"Table successfully deleted with name --{self.table_name}--\n")
        except Exception as e:
            print(f"Error occurred during deleting table with name --{self.table_name}--\nError message:\n{e}\n")


    # function for inserting data
    def insert_data(
            self,
            phone_number: str, 
            person_name: str, 
            appointment_date: str, 
            appointment_time: str, 
            age: int = None):
        # Calculate appointment_end_time (5 minutes after appointment_time)
        appointment_time_obj = datetime.strptime(appointment_time, "%H:%M:%S")
        appointment_end_time_obj = appointment_time_obj + timedelta(minutes=5)
        appointment_end_time = appointment_end_time_obj.strftime("%H:%M:%S")

        # SQL query to insert data into the table
        insert_query = f'''
        INSERT INTO {self.table_name} (phone_number, person_name, age, appointment_date, appointment_time, appointment_end_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        '''
        
        # Data to insert
        data = (phone_number, person_name, age, appointment_date, appointment_time, appointment_end_time)
        
        try:
            conn, cursor = self.__get_connection_cursor()
            # Execute the query and commit the transaction
            cursor.execute(insert_query, data)
            conn.commit()

            if conn.is_connected():
                cursor.close()
                conn.close()
            print(f"Data inserted successfully into the table.")
        except Exception as e:
            print(f"Error occurred during insertion.\nError message:\n{e}\n")


    # function for searching data in table
    def search_data(self, phone_number):
        conn, cursor = self.__get_connection_cursor()
        result = None
        try:
            search_query = f"SELECT * FROM {self.table_name} WHERE phone_number = %s"   # sql query for searching data
            cursor.execute(search_query, (phone_number,))
            result = cursor.fetchone()
            if conn.is_connected():
                cursor.close()
                conn.close()
        except Exception as e:
            print(f"Error occurred while searching data.\nError Message:\n{e}\n")
        return result
    

    # function for updating data
    def update_data(self, phone_number: str, person_name: str = None, age: int = None, appointment_date: str = None, appointment_time: str = None):
        try:
            # establishing connection to the database
            conn, cursor = self.__get_connection_cursor()
            
            # Check if the phone number exists
            search_query = f"SELECT * FROM {self.table_name} WHERE phone_number = %s"    # sql query for searching data
            cursor.execute(search_query, (phone_number,))
            result = cursor.fetchone()
            
            if result:  # when there is a data exist to update
                update_fields = []
                update_values = []
                
                if person_name:
                    update_fields.append("person_name = %s")
                    update_values.append(person_name)
                
                if age is not None:
                    update_fields.append("age = %s")
                    update_values.append(age)
                
                if appointment_date:
                    # Check if the date is in 'day-month-year' format and convert it if needed
                    try:
                        # Try to convert from 'DD-MM-YYYY' format to 'YYYY-MM-DD' format
                        appointment_date_obj = datetime.strptime(appointment_date, "%d-%m-%Y")
                        formatted_appointment_date = appointment_date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        # If it's already in 'YYYY-MM-DD', use it directly
                        formatted_appointment_date = appointment_date

                    update_fields.append("appointment_date = %s")
                    update_values.append(formatted_appointment_date)
                
                if appointment_time:
                    # Handle both 'HH:MM' and 'HH:MM:SS' formats by slicing to 'HH:MM'
                    try:
                        appointment_time_obj = datetime.strptime(appointment_time[:5], "%H:%M")
                    except ValueError:
                        print(f"Invalid time format: {appointment_time}")
                        return

                    # Appointment end time is 5 minutes later
                    appointment_end_time_obj = appointment_time_obj + timedelta(minutes=5)
                    formatted_appointment_time = appointment_time_obj.strftime("%H:%M:%S")
                    formatted_appointment_end_time = appointment_end_time_obj.strftime("%H:%M:%S")
                    
                    update_fields.append("appointment_time = %s")
                    update_values.append(formatted_appointment_time)
                    
                    update_fields.append("appointment_end_time = %s")
                    update_values.append(formatted_appointment_end_time)
                
                # Only update if there are fields to update
                if update_fields:
                    update_query = f"UPDATE {self.table_name} SET {', '.join(update_fields)} WHERE phone_number = %s"
                    update_values.append(phone_number)  # Add the phone_number to the end for the WHERE clause
                    cursor.execute(update_query, tuple(update_values))
                    conn.commit()
                    
                    print(f"Data for phone number --{phone_number}-- updated successfully.")
                else:
                    print("No new information provided to update.")
            else:
                print(f"No data found for phone number {phone_number}. Update not performed.")
        except Exception as e:
            print(f"Error occurred while updating.\nError message:\n{e}\n")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()


    # function for deleting data from the database
    def delete_data(self, phone_number:str):
        try:
            conn, cursor = self.__get_connection_cursor()
            result = self.__format_search_result(self.search_data(phone_number=phone_number))
            print(result)
            if result:
                delete_query = f"DELETE FROM {self.table_name} WHERE phone_number = %s"    # sql query for deleting data from table
                cursor.execute(delete_query, (phone_number,))
                conn.commit()
                print(f"Data deleted from the table with phone number {phone_number}\n")
            else:
                print(f"Data with this phone number does not exists.")
        except Exception as e:
            print(f"Error occurred while deleting data,\nError message:\n{e}\n")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()


    # function for deleting entire database
    def delete_database(self, database_name:str):
        try:
            conn, cursor = self.__get_connection_cursor()
            cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")    # sql query for deleting database
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error occurred during deleting database with name --{database_name}--\nError message:\n{e}\n")
        else:
            print(f"Database is deleted with the name --{database_name}--\n")


if __name__ == "__main__":
    db = MySQLDatabase(host="localhost",
                       user="root",
                       password="",
                       database_name="db",
                       table_name="mytable")

    # data = {
    #     "table_name": "mytable", 
    #     "phone_number": "01234567890",
    #     "person_name": "John Doe",
    #     "appointment_date": "2024-10-01",
    #     "appointment_time": "10:00:00"
    # }

    # db.insert_data(**data)
    # db.insert_data(phone_number="01234567890", person_name="John Doe", appointment_date="2024-10-01", appointment_time="10:00:00")
    # db.insert_data(phone_number="09876543210", person_name="Jane Smith", appointment_date="2024-10-02", appointment_time="11:00:00", age=25)

    # db.delete_data(phone_number="09876543210")
    
    # db.update_data(phone_number="09876543210")
    # db.update_data(phone_number="09876543210", person_name="Muntasir Ahmed")
