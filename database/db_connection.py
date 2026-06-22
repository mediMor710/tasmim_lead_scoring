import psycopg2
import os

def get_connection():
    """
    Here we set the connection to the database,
    so if we in case of changing the informations of PostgreSQL,
    we can edit it from here
    """

    host = os.environ.get('DB_HOST', 'localhost')
    conn = psycopg2.connect(
        host = 'localhost',
        database='tasmimdb',
        user='postgres',
        password='Medi@@710',
        port = 5432,
        connect_timeout = 10
    )
    return conn