from database import Base, engine
from models import *

def create_tables():
    Base.metadata.create_all(engine)
    print("All tables created successfully")

if __name__ == "__main__":
    create_tables()