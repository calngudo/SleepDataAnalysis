import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = f"postgresql+psycopg2://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'sleep_analysis')}"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class SleepRecord(Base):
    __tablename__ = 'sleep_records'
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50))
    date = Column(DateTime)
    avg_score = Column(Float)
    stress = Column(Float)
    avg_resting_heart_rate = Column(Float)
    avg_high_heart_rate = Column(Float)
    steps = Column(Integer)
    intensity_minutes = Column(Integer)
    avg_sleep_need = Column(Float)
    avg_duration = Column(Float)
    sleep_duration = Column(Float)
    created_at = Column(DateTime)

Base.metadata.create_all(engine)

def insert_sleep_record(record):
    session = Session()
    sleep_record = SleepRecord(
        user_id=record['user_id'],
        date=record['date'],
        avg_score=record['avg_score'],
        stress=record['stress'],
        avg_resting_heart_rate=record['avg_resting_heart_rate'],
        avg_high_heart_rate=record['avg_high_heart_rate'],
        steps=record['steps'],
        intensity_minutes=record['intensity_minutes'],
        avg_sleep_need=record['avg_sleep_need'],
        avg_duration=record['avg_duration'],
        sleep_duration=record['sleep_duration'],
        created_at=datetime.utcnow()
    )
    session.add(sleep_record)
    session.commit()
    session.close()