from app import init_database, predictor, insert_customers_to_db
init_database()
df = predictor.generate_sample_data(500)
insert_customers_to_db(df)
