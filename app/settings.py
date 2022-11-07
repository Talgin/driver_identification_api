import os
from pathlib import Path

#ip = os.environ['FASTAPI_IP']                                               #Default IP address of the server
#port = os.environ['FASTAPI_PORT']                                           #Default port of server
#protocol = os.environ['FASTAPI_PROTOCOL']                                   #Default protocol
models_dir = os.environ['FASTAPI_MODELS_DIR']
models_threshold = float(os.environ['FASTAPI_MODELS_THRESHOLD'])             #Face detection threshold

# DATA FOLDERS
csv_folder = os.environ['FASTAPI_CSV_FOLDER']                                        #/CARSHARING/STORAGE/csv

# POSTGRES DB RELATED VARIABLES
pg_server = os.environ['FASTAPI_PG_SERVER']                                 #Postgresdb server ip address
pg_port = os.environ['FASTAPI_PG_PORT']                                     #Postgresdb server default port
pg_db = os.environ['FASTAPI_PG_DB']                                         #Postgresdb database name
pg_username = os.environ['FASTAPI_PG_USER']                                 #Postgresdb username
pg_password = os.environ['FASTAPI_PG_PASS']                                 #Postgresdb password

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

# Postgresql connection settings: host, port, dbname, user, pwd
PG_CONNECTION = [pg_server, pg_port, pg_db, pg_username, pg_password]