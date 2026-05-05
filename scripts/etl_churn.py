import pandas as pd
import sqlite3

# CONFIG
CSV_PATH = '/workspaces/hackathon-churn-prediction/dados/Bank_Customer_Churn_Prediction.csv'
DB_PATH = "churn.db"

# CONEXÃO
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# CRIAR TABELAS
cursor.executescript("""
DROP TABLE IF EXISTS fact_customer_behavior;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_country;
DROP TABLE IF EXISTS dim_gender;

CREATE TABLE dim_customer (
    customer_id INTEGER PRIMARY KEY,
    credit_score INTEGER,
    age INTEGER,
    tenure INTEGER,
    estimated_salary REAL
);

CREATE TABLE dim_country (
    country_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_name TEXT UNIQUE
);

CREATE TABLE dim_gender (
    gender_id INTEGER PRIMARY KEY AUTOINCREMENT,
    gender TEXT UNIQUE
);

CREATE TABLE fact_customer_behavior (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    country_id INTEGER,
    gender_id INTEGER,
    balance REAL,
    products_number INTEGER,
    credit_card INTEGER,
    active_member INTEGER,
    churn INTEGER,

    FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id),
    FOREIGN KEY (country_id) REFERENCES dim_country(country_id),
    FOREIGN KEY (gender_id) REFERENCES dim_gender(gender_id)
);
""")

conn.commit()

# EXTRAÇÃO
df = pd.read_csv(CSV_PATH)

# TRANSFORMAÇÃO

# padronizar nomes
df.columns = df.columns.str.lower()

# remover espaços
df['country'] = df['country'].str.strip()
df['gender'] = df['gender'].str.strip()

# remover duplicados
df = df.drop_duplicates()

# DIMENSÕES

# COUNTRY
countries = df[['country']].drop_duplicates().sort_values('country')
countries.to_sql('dim_country', conn, if_exists='append', index=False)

# GENDER
genders = df[['gender']].drop_duplicates().sort_values('gender')
genders.to_sql('dim_gender', conn, if_exists='append', index=False)

# MAPEAMENTO DE IDS

country_map = pd.read_sql("SELECT * FROM dim_country", conn)
gender_map = pd.read_sql("SELECT * FROM dim_gender", conn)

df = df.merge(country_map, left_on='country', right_on='country_name', how='left')
df = df.merge(gender_map, on='gender', how='left')

# DIM CUSTOMER

dim_customer = df[[
    'customer_id',
    'credit_score',
    'age',
    'tenure',
    'estimated_salary'
]].drop_duplicates()

dim_customer.to_sql('dim_customer', conn, if_exists='append', index=False)

# FATO

fact = df[[
    'customer_id',
    'country_id',
    'gender_id',
    'balance',
    'products_number',
    'credit_card',
    'active_member',
    'churn'
]]

fact.to_sql('fact_customer_behavior', conn, if_exists='append', index=False)

# ÍNDICES (performance)

cursor.executescript("""
CREATE INDEX idx_fact_customer ON fact_customer_behavior(customer_id);
CREATE INDEX idx_fact_country ON fact_customer_behavior(country_id);
CREATE INDEX idx_fact_gender ON fact_customer_behavior(gender_id);
""")

conn.commit()
conn.close()

print("ETL finalizado com sucesso!")