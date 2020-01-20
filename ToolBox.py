from sqlalchemy import create_engine
def WriteToMySQL(data,table_name,db_name):
    engine = create_engine('mysql+mysqlconnector://root:1234567890@localhost:3306/'+db_name)
    data.to_sql(name=table_name,con=engine,index=False,if_exists='append')
    return None


