from database import Base
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, ForeignKey, Text, BigInteger, JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime


# =========================
# TABLE: players
# =========================
class Player(Base):
    __tablename__ = "players"

    id = Column(String(50), primary_key=True)
    name = Column(String(255))
    team = Column(String(255))
    position = Column(String(50))
    age = Column(Integer)
    nationality = Column(String(100))
    height_cm = Column(Integer)
    weight_kg = Column(Integer)
    preferred_foot = Column(String(20))
    photo = Column(String(255))

    create_at = Column(DateTime, default=datetime.now)
    update_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    stats = relationship("PlayerStats", back_populates="player", cascade="all, delete")


# =========================
# TABLE: player_stats
# =========================
class PlayerStats(Base):
    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"))

    season = Column(String(20))
    appearances = Column(Integer)
    goals = Column(Integer)
    assists = Column(Integer)
    minutes_played = Column(Integer)
    rating = Column(Float)
    key_passes = Column(Integer)

    create_at = Column(DateTime, default=datetime.now)
    update_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    player = relationship("Player", back_populates="stats")


# =========================
# TABLE: model1_attributes
# =========================
class Model1Attributes(Base):
    __tablename__ = "model1_attributes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"))

    avg_speed_kmh = Column(Float)
    max_speed_kmh = Column(Float)
    sprint_speed_kmh = Column(Float)
    total_distance_m = Column(Float)
    stamina_percentage = Column(Float)
    acceleration = Column(Float)
    deceleration = Column(Float)
    agility_score = Column(Float)
    jump_count = Column(Integer)

    normalized_attribute = Column(JSON)

    create_at = Column(DateTime, default=datetime.now)


# =========================
# TABLE: model2_results
# =========================
class Model2Results(Base):
    __tablename__ = "model2_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"))

    recommended_position = Column(String(50))
    secondary_position = Column(String(50))
    potential_score = Column(Integer)
    behavior_cluster = Column(String(100))

    explainability = Column(JSON)
    model_version = Column(String(50))

    create_at = Column(DateTime, default=datetime.now)


# =========================
# TABLE: model2_feature_fusios
# =========================
class Model2FeatureFusion(Base):
    __tablename__ = "model2_feature_fusios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"))

    fused_features = Column(JSON)
    model1_source = Column(JSON)
    stats_source = Column(JSON)
    attributes_source = Column(JSON)

    created_at = Column(DateTime, default=datetime.now)


# =========================
# TABLE: ai_analysis
# =========================
class AIAnalysis(Base):
    __tablename__ = "ai_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"))

    overall_score = Column(Integer)
    strengths = Column(Text)
    weakness = Column(Text)
    recommended_role = Column(String(100))
    market_value = Column(BigInteger)
    summary_report = Column(Text)

    create_at = Column(DateTime, default=datetime.now)