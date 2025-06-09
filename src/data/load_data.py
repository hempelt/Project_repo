import pandas as pd
from sqlalchemy import create_engine

# Connection parameters for PostgreSQL
user = "postgres"
password = "monitor"
host = "localhost"
port = "5432"
database = "LIMS"

# Create PostgreSQL connection string
connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

# Create SQLAlchemy engine
engine = create_engine(connection_string)

# Load data into DataFrame
query = "SELECT * FROM formulation_screening"
df = pd.read_sql_query(query, engine)

# Close the database connection
engine.dispose()

# Export DataFrame to CSV
output_csv_path = r"C:\Users\hempe\Studium\Real_Project\Project_repo\data\raw\raw_data.csv"
df.to_csv(output_csv_path, index=False) 

