from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, LargeBinary, String, Text
from sqlalchemy.orm import deferred, relationship

from backend.skinvision.database import Base


class User(Base):
	__tablename__ = "users"

	id = Column(Integer, primary_key=True, index=True)
	username = Column(String(20), unique=True, nullable=False, index=True)
	password_hash = Column(String(255), nullable=False)
	email = Column(String(255), unique=True, index=True, nullable=True)
	role = Column(String(20), default="doctor", nullable=True)
	is_active = Column(Boolean, default=True, nullable=False)
	last_login = Column(DateTime, nullable=True)
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

	tasks = relationship("AnalysisTask", back_populates="user", cascade="all, delete-orphan")


class AnalysisTask(Base):
	__tablename__ = "analysis_tasks"

	id = Column(Integer, primary_key=True, index=True)
	user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
	patient_age = Column(Integer, nullable=True)
	patient_symptoms = Column(String(500), nullable=True)
	patient_duration = Column(String(100), nullable=True)
	patient_id = Column(String(30), nullable=True)
	pipeline_mode = Column(String(50), nullable=True)
	inference_time_ms = Column(Float, nullable=True)
	area_ratio = Column(Float, nullable=True)
	irregularity_index = Column(Float, nullable=True)
	color_variance = Column(Float, nullable=True)
	features_json = Column(JSON, nullable=True)
	mask_path = Column(String(255), nullable=True)
	roi_path = Column(String(255), nullable=True)
	gradcam_path = Column(String(255), nullable=True)
	probability_path = Column(String(255), nullable=True)
	overlay_path = Column(String(255), nullable=True)
	classification_json = Column(JSON, nullable=True)
	reliability_json = Column(JSON, nullable=True)
	model_info_json = Column(JSON, nullable=True)
	report_status = Column(String(50), nullable=True)
	report_metadata_json = Column(JSON, nullable=True)
	status = Column(String(50), default="processing", nullable=False)
	error_message = Column(Text, nullable=True)
	llm_report = Column(Text, nullable=True)
	edited_report = Column(Text, nullable=True)
	report_version = Column(Integer, default=0, server_default="0", nullable=False)
	report_edited_at = Column(DateTime, nullable=True)
	original_image = deferred(Column(LargeBinary, nullable=True))
	original_media_type = Column(String(50), nullable=True)
	request_id = Column(String(100), unique=True, index=True, nullable=True)
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

	user = relationship("User", back_populates="tasks")


class AuditLog(Base):
	__tablename__ = "audit_logs"

	id = Column(Integer, primary_key=True, index=True)
	user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
	action = Column(String(100), nullable=True)
	request_id = Column(String(100), index=True, nullable=True)
	detail_json = Column(JSON, nullable=True)
	ip_address = Column(String(45), nullable=True)
	created_at = Column(DateTime, default=datetime.utcnow)
