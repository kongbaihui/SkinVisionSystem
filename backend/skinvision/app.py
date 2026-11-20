import json
import math
import os
import re
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager, aclosing
from datetime import datetime, timedelta
from typing import Generator, Optional, Literal

import cv2
import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import func, update
from sqlalchemy.orm import Session, sessionmaker

from backend.skinvision.runtime import get_default_temp_dir, load_project_env, resolve_path

load_project_env()

from backend.skinvision.auth import (
	ACCESS_TOKEN_EXPIRE_SECONDS,
	REFRESH_TOKEN_EXPIRE_SECONDS,
	bearer_scheme,
	create_access_token,
	create_refresh_token,
	decode_token,
	get_current_user_id,
	hash_password,
	require_admin,
	verify_password,
)
from backend.skinvision.database import Base, engine, get_db
from backend.skinvision.db_models import AnalysisTask, AuditLog, User
from backend.skinvision.llm_gateway import LLMGateway
from backend.skinvision.pipeline import SkinVisionPipeline
from backend.skinvision.ai.classification import Classifier
from backend.skinvision.analysis_results import extra_fields, task_features
from backend.skinvision.rule_cues import RISK_SOURCE, rule_level
from backend.skinvision.image_io import read_image
from backend.skinvision import auth
from backend.skinvision.analysis_workflow import segment, additional_results, report_events, interrupt_pending
from backend.skinvision.streaming import ClosingStreamingResponse


def _load_pipeline_from_env():
	weights_path = resolve_path(os.getenv("SKINVISION_WEIGHTS", ""))
	threshold = float(os.getenv("SKINVISION_THRESHOLD", "0.5"))
	use_cpu = os.getenv("SKINVISION_CPU", "0") == "1"
	return SkinVisionPipeline(weights_path=weights_path, threshold=threshold, use_cpu=use_cpu,
	                          normalization=os.getenv('SKINVISION_NORMALIZATION', 'rgb_minus1_1'))


pipeline = _load_pipeline_from_env()
llm_api_key = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY", "")).strip()
llm_model = os.getenv("SKINVISION_LLM_MODEL", "qwen3-vl-plus").strip()
llm_timeout = float(os.getenv("SKINVISION_LLM_TIMEOUT", "60"))
llm_gateway = LLMGateway(api_key=llm_api_key, model=llm_model, timeout=llm_timeout) if llm_api_key else None

classifier = Classifier(llm_api_key, model=os.getenv("SKINVISION_CLASSIFICATION_MODEL", llm_model), timeout=llm_timeout)
if os.getenv('SKINVISION_CLASSIFICATION_BACKEND', 'api') == 'local':
	from backend.skinvision.ai.local_classification import LocalClassifier
	classifier = LocalClassifier(resolve_path(os.getenv('SKINVISION_CLASSIFICATION_DIR', '')),
	                             device=os.getenv('SKINVISION_CLASSIFICATION_DEVICE', 'cpu'))

TEMP_DIR = resolve_path(os.getenv("SKINVISION_TEMP_DIR", get_default_temp_dir()))
MAX_FILE_SIZE = int(os.getenv("SKINVISION_MAX_FILE_SIZE", str(50 * 1024 * 1024)))
FILE_CLEANUP_HOURS = float(os.getenv("SKINVISION_FILE_CLEANUP_HOURS", "24"))

os.makedirs(TEMP_DIR, exist_ok=True)


def _task_artifact_paths(task):
    reliability = task.reliability_json or {}
    return [task.mask_path, task.roi_path, task.gradcam_path, task.overlay_path, task.probability_path,
            *[reliability.get(key) for key in ['variance_path', 'variance_image_path', 'occlusion_path', 'occlusion_image_path']]]


def _cleanup_old_files() -> int:
    # Historical output is durable. Only remove unreferenced generated files.
    cutoff = time.time() - FILE_CLEANUP_HOURS * 3600
    removed = 0
    try:
        with Session(bind=engine) as session:
            referenced = {os.path.basename(path) for task in session.query(AnalysisTask).all()
                          for path in _task_artifact_paths(task) if path}
        for filename in os.listdir(TEMP_DIR):
            if filename in referenced or not re.fullmatch(r'(mask|roi|gradcam|overlay|prob|variance|occlusion)_[0-9a-f-]+\.(png|npy)', filename):
                continue
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff:
                try:
                    os.remove(file_path)
                    removed += 1
                except OSError:
                    pass
    except Exception:
        # If the database cannot be checked, keep all files.
        pass
    return removed


def _start_cleanup_thread() -> None:
	def _loop():
		while True:
			_cleanup_old_files()
			time.sleep(3600)

	threading.Thread(target=_loop, daemon=True, name="file-cleanup").start()


_MAX_LOGIN_ATTEMPTS = int(os.getenv("SKINVISION_MAX_LOGIN_ATTEMPTS", "5"))
_LOGIN_LOCKOUT_SECONDS = int(os.getenv("SKINVISION_LOGIN_LOCKOUT_SECONDS", "900"))
_LOGIN_ATTEMPTS: dict = {}
_LOGIN_LOCK = threading.Lock()


def _check_login_locked(username: str) -> None:
	now = datetime.utcnow()
	with _LOGIN_LOCK:
		entry = _LOGIN_ATTEMPTS.get(username)
		if entry and entry["locked_until"] and now < entry["locked_until"]:
			remaining = int((entry["locked_until"] - now).total_seconds())
			raise HTTPException(
				status_code=429,
				detail={
					"status_code": 429,
					"error": "too_many_attempts",
					"message": f"登录失败次数过多，请 {remaining} 秒后重试",
				},
			)


def _record_login_failure(username: str) -> None:
	now = datetime.utcnow()
	with _LOGIN_LOCK:
		entry = _LOGIN_ATTEMPTS.get(username, {"count": 0, "locked_until": None})
		if entry["locked_until"] and now >= entry["locked_until"]:
			entry = {"count": 0, "locked_until": None}
		entry["count"] += 1
		if entry["count"] >= _MAX_LOGIN_ATTEMPTS:
			entry["locked_until"] = now + timedelta(seconds=_LOGIN_LOCKOUT_SECONDS)
			entry["count"] = 0
		_LOGIN_ATTEMPTS[username] = entry


def _reset_login_attempts(username: str) -> None:
	with _LOGIN_LOCK:
		_LOGIN_ATTEMPTS.pop(username, None)


_CORS_ORIGINS_RAW = os.getenv("SKINVISION_CORS_ORIGINS", "*")
_CORS_ORIGINS = ["*"] if _CORS_ORIGINS_RAW.strip() == "*" else [origin.strip() for origin in _CORS_ORIGINS_RAW.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(_: FastAPI):
	Base.metadata.create_all(bind=engine)
	_start_cleanup_thread()
	yield


app = FastAPI(title="SkinVision Gateway", version="1.0.0", lifespan=lifespan)

app.add_middleware(
	CORSMiddleware,
	allow_origins=_CORS_ORIGINS,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
app.mount("/static/outputs", StaticFiles(directory=TEMP_DIR), name="static_outputs")


class RegisterRequest(BaseModel):
	username: str = Field(..., min_length=3, max_length=20, pattern=r"^[a-zA-Z0-9_]+$")
	password: str = Field(..., min_length=8)
	email: Optional[EmailStr] = None

	@field_validator("password")
	@classmethod
	def password_complexity(cls, value: str) -> str:
		if not re.search(r"[A-Za-z]", value) or not re.search(r"[0-9]", value):
			raise ValueError("密码需同时包含字母和数字")
		return value


class LoginRequest(BaseModel):
	username: str
	password: str


class RefreshRequest(BaseModel):
	refresh_token: str


class LogoutRequest(BaseModel):
	refresh_token: Optional[str] = None


class ReportRevision(BaseModel):
	text: str = Field(min_length=1, max_length=200000)
	version: int = Field(ge=0)

	@field_validator("text")
	@classmethod
	def check_text(cls, value):
		if not value.strip():
			raise ValueError("报告正文不能为空")
		return value


def _metric_text(features, name, precision, unit=""):
	value = features.get(name)
	return f"{value:.{precision}f}{unit}" if value is not None else "未记录"


def _build_markdown_report(features, patient_age, symptoms, duration):
	risk = rule_level(features.get("irregularity_index"))

	return [
		"### 影像特征分析\n",
		f"- 边缘不规则指数：{_metric_text(features, 'irregularity_index', 4)}\n",
		f"- 病灶面积占比：{_metric_text(features, 'area_ratio', 4, '%')}\n",
		f"- 色彩变异度：{_metric_text(features, 'color_variance', 6)}\n",
		f"\n### 既有规则提示\n- 来源：{RISK_SOURCE}\n",
		f"- 规则提示（非疾病分类）：**{risk}**\n",
		f"- 患者信息：年龄 {patient_age if patient_age is not None else '未提供'}，症状 {symptoms}，病程 {duration}\n",
		"\n### 临床处置建议\n",
		"- 建议结合皮肤镜复查与专科医生面诊，必要时进行病理检查。\n",
		"\n---\n*本报告仅供临床参考，不替代专科医生诊断意见。*\n",
	]


def _sse_pack(event_type: str, data: dict) -> str:
	return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _write_audit(
	db: Session,
	action: str,
	user_id: Optional[int] = None,
	request_id: Optional[str] = None,
	detail: Optional[dict] = None,
	ip: Optional[str] = None,
):
	log = AuditLog(user_id=user_id, action=action, request_id=request_id, detail_json=detail or {}, ip_address=ip)
	db.add(log)
	db.commit()


def _get_client_ip(request: Request) -> str:
	forwarded = request.headers.get("X-Forwarded-For")
	if forwarded:
		return forwarded.split(",")[0].strip()
	return request.client.host if request.client else ""


@app.get("/health")
def health_check():
	return {"status": "ok", "pipeline_mode": pipeline.mode, "model_load_error": pipeline.load_error,
			"device": str(pipeline.device), "auth_enabled": auth.ENABLE_AUTH}


@app.post("/api/v1/auth/register", status_code=201)
def register(body: RegisterRequest, request: Request, db: Session = Depends(get_db)):
	if db.query(User).filter(User.username == body.username).first():
		raise HTTPException(status_code=400, detail={"status_code": 400, "error": "username_already_exists", "message": "用户名已被注册"})
	if body.email and db.query(User).filter(User.email == body.email).first():
		raise HTTPException(status_code=400, detail={"status_code": 400, "error": "email_already_exists", "message": "邮箱已被注册"})

	user = User(username=body.username, password_hash=hash_password(body.password), email=body.email)
	db.add(user)
	db.commit()
	db.refresh(user)

	_write_audit(db, action="register", user_id=user.id, detail={"username": user.username}, ip=_get_client_ip(request))
	return {
		"id": user.id,
		"username": user.username,
		"email": user.email,
		"created_at": user.created_at.isoformat() + "Z",
		"access_token": create_access_token(user.id, user.username),
		"refresh_token": create_refresh_token(user.id),
		"token_type": "Bearer",
		"expires_in": ACCESS_TOKEN_EXPIRE_SECONDS,
	}


@app.post("/api/v1/auth/login")
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
	_check_login_locked(body.username)
	user = db.query(User).filter(User.username == body.username).first()
	if not user or not verify_password(body.password, user.password_hash):
		_record_login_failure(body.username)
		_write_audit(db, action="login_failed", detail={"username": body.username}, ip=_get_client_ip(request))
		raise HTTPException(status_code=401, detail={"status_code": 401, "error": "invalid_credentials", "message": "用户名或密码错误"})
	if not user.is_active:
		raise HTTPException(status_code=401, detail={"status_code": 401, "error": "account_disabled", "message": "账户已被禁用"})

	_reset_login_attempts(body.username)
	user.last_login = datetime.utcnow()
	db.commit()
	_write_audit(db, action="login", user_id=user.id, detail={"username": user.username}, ip=_get_client_ip(request))
	return {
		"user": {"id": user.id, "username": user.username, "email": user.email, "role": user.role or "doctor"},
		"access_token": create_access_token(user.id, user.username),
		"refresh_token": create_refresh_token(user.id),
		"token_type": "Bearer",
		"expires_in": ACCESS_TOKEN_EXPIRE_SECONDS,
		"refresh_expires_in": REFRESH_TOKEN_EXPIRE_SECONDS,
	}


@app.post("/api/v1/auth/refresh")
def refresh_token(body: RefreshRequest, db: Session = Depends(get_db)):
	payload = decode_token(body.refresh_token, expected_type="refresh")
	user_id = int(payload["sub"])
	user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
	if not user:
		raise HTTPException(status_code=401, detail={"status_code": 401, "error": "unauthorized", "message": "用户不存在或已禁用"})
	return {
		"access_token": create_access_token(user.id, user.username),
		"token_type": "Bearer",
		"expires_in": ACCESS_TOKEN_EXPIRE_SECONDS,
	}


@app.get("/api/v1/auth/me")
def me(credentials=Depends(bearer_scheme), user_id: Optional[int] = Depends(get_current_user_id), db: Session = Depends(get_db)):
	if user_id is None:
		return {"message": "认证未启用"}

	user = db.query(User).filter(User.id == user_id).first()
	if not user:
		raise HTTPException(status_code=404, detail="用户不存在")
	return {
		"id": user.id,
		"username": user.username,
		"email": user.email,
		"role": user.role or "doctor",
		"is_active": user.is_active,
		"created_at": user.created_at.isoformat() + "Z",
		"updated_at": user.updated_at.isoformat() + "Z" if user.updated_at else None,
	}


@app.post("/api/v1/auth/logout")
def logout(body: LogoutRequest, request: Request, user_id: Optional[int] = Depends(get_current_user_id), db: Session = Depends(get_db)):
	_write_audit(db, action="logout", user_id=user_id, ip=_get_client_ip(request))
	return {"message": "成功退出登录"}


async def _read_analysis_image(image):
    content = await image.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=422, detail=f"图片大小超出限制（{MAX_FILE_SIZE // 1024 // 1024} MB）")
    image_bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) if content else None
    if image_bgr is None:
        return None, None, None
    # Identify the format from the decoded upload independently of its declared MIME.
    try:
        with Image.open(BytesIO(content)) as source:
            media_type = Image.MIME.get(source.format)
    except (OSError, ValueError):
        media_type = None
    if media_type not in {'image/png', 'image/jpeg', 'image/webp', 'image/bmp', 'image/tiff', 'image/gif'}:
        content = cv2.imencode('.png', image_bgr)[1].tobytes()
        media_type = 'image/png'
    return image_bgr, content, media_type


def _analysis_response(task):
    return {
        "id": task.id, "user_id": task.user_id, "patient_id": task.patient_id, "pipeline_mode": task.pipeline_mode,
        "patient": {"patient_age": task.patient_age, "symptoms": task.patient_symptoms,
                    "duration": task.patient_duration},
        "features": task.features_json, "mask_path": task.mask_path, "roi_path": task.roi_path,
        "gradcam_url": task.gradcam_path, "inference_time_ms": task.inference_time_ms,
        "created_at": task.created_at.isoformat() + "Z", "status": task.status, **extra_fields(task),
    }


@app.post("/api/v1/diagnostic/analyze")
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    patient_age: Optional[int] = Form(None, ge=1, le=150),
    symptoms: str = Form("未提供", max_length=500),
    duration: str = Form("未提供"),
    patient_id: Optional[str] = Form(None, max_length=30),
    classification_view: Literal['full', 'roi'] = Form('full'),
    include_reliability: bool = Form(True),
    include_occlusion: bool = Form(False),
    user_id: Optional[int] = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    image_bgr, original, media_type = await _read_analysis_image(image)
    if image_bgr is None:
        return JSONResponse(status_code=400, content={'error': 'invalid image file'})
    factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    task, result = await segment(factory, pipeline, image_bgr, TEMP_DIR, user_id=user_id,
                                 original_image=original, original_media_type=media_type,
                                 patient_age=patient_age, patient_symptoms=symptoms,
                                 patient_duration=duration, patient_id=patient_id, request_id=str(uuid.uuid4()))
    if result is None:
        raise HTTPException(status_code=500, detail=task.error_message)
    async for _ in additional_results(factory, task, result, image_bgr, TEMP_DIR, pipeline, classifier,
                                       classification_view, include_reliability, include_occlusion):
        pass
    return _analysis_response(task)


@app.post("/api/v1/diagnostic/analyze_stream")
async def analyze_stream(
    request: Request,
    image: UploadFile = File(...),
    patient_age: Optional[int] = Form(None, ge=1, le=150),
    symptoms: str = Form("未提供", max_length=500),
    duration: str = Form("未提供"),
    patient_id: Optional[str] = Form(None, max_length=30),
    classification_view: Literal['full', 'roi'] = Form('full'),
    include_reliability: bool = Form(True),
    include_occlusion: bool = Form(False),
    report_variant: Literal['text_only', 'image_text', 'constrained'] = Form('constrained'),
    user_id: Optional[int] = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    image_bgr, original, media_type = await _read_analysis_image(image)
    if image_bgr is None:
        return JSONResponse(status_code=400, content={'error': 'invalid image file'})
    factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    task, result = await segment(factory, pipeline, image_bgr, TEMP_DIR, user_id=user_id,
                                 original_image=original, original_media_type=media_type,
                                 patient_age=patient_age, patient_symptoms=symptoms,
                                 patient_duration=duration, patient_id=patient_id, request_id=str(uuid.uuid4()))

    async def events():
        if result is None:
            yield _sse_pack('error', {'module': 'segmentation', 'message': task.error_message, 'task_id': task.id})
            yield _sse_pack('done', {'status': 'finished', 'task_id': task.id})
            return
        try:
            yield _sse_pack('meta', {**_analysis_response(task), 'task_id': task.id,
                                   'format': 'markdown-chunks', 'llm_enabled': llm_gateway is not None})
            async with aclosing(additional_results(factory, task, result, image_bgr, TEMP_DIR,
                                                   pipeline, classifier, classification_view,
                                                   include_reliability, include_occlusion)) as additions:
                async for event, data in additions:
                    yield _sse_pack(event, data)
            lines = _build_markdown_report(result['features'], patient_age, symptoms, duration)
            async with aclosing(report_events(factory, task, result, llm_gateway, report_variant, lines)) as report:
                async for event, data in report:
                    yield _sse_pack(event, data)
        finally:
            interrupt_pending(factory, task.id)
        yield _sse_pack('done', {'status': 'finished', 'task_id': task.id})

    return ClosingStreamingResponse(events(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
                             on_finalize=lambda: interrupt_pending(factory, task.id))


@app.get("/api/v1/tasks")
def get_tasks(
	page: int = Query(1, ge=1),
	page_size: int = Query(10, ge=1, le=100),
	status: Optional[str] = Query(None),
	sort: str = Query("created_at.desc"),
	user_id: Optional[int] = Depends(get_current_user_id),
	db: Session = Depends(get_db),
):
	query = db.query(AnalysisTask)
	query = query.filter(AnalysisTask.user_id == user_id)
	if status:
		query = query.filter(AnalysisTask.status == status)

	sortable_fields = {
		"created_at": AnalysisTask.created_at,
		"updated_at": AnalysisTask.updated_at,
		"irregularity_index": AnalysisTask.irregularity_index,
		"area_ratio": AnalysisTask.area_ratio,
		"inference_time_ms": AnalysisTask.inference_time_ms,
		"patient_age": AnalysisTask.patient_age,
		"id": AnalysisTask.id,
	}
	sort_field, _, sort_dir = sort.partition(".")
	column = sortable_fields.get(sort_field, AnalysisTask.created_at)
	query = query.order_by(column.asc() if sort_dir == "asc" else column.desc())

	total_count = query.count()
	total_pages = math.ceil(total_count / page_size) if page_size else 1
	items = query.offset((page - 1) * page_size).limit(page_size).all()
	return {
		"data": [
			{
				"id": task.id,
				"patient_id": task.patient_id,
				"patient_age": task.patient_age,
				"patient_symptoms": task.patient_symptoms,
				"patient_duration": task.patient_duration,
				"irregularity_index": task.irregularity_index,
				"area_ratio": task.area_ratio,
				**extra_fields(task),
				"status": task.status,
				"created_at": task.created_at.isoformat() + "Z" if task.created_at else None,
				"updated_at": task.updated_at.isoformat() + "Z" if task.updated_at else None,
			}
			for task in items
		],
		"pagination": {"page": page, "page_size": page_size, "total_count": total_count, "total_pages": total_pages},
	}


def _authorized_task(db, task_id, user_id, *, require_identity=False):
	if require_identity and user_id is None:
		raise HTTPException(status_code=401, detail="需要登录")
	task = db.get(AnalysisTask, task_id)
	if not task:
		raise HTTPException(status_code=404, detail={"status_code": 404, "error": "not_found", "message": "任务不存在"})
	if task.user_id == user_id:
		return task
	user = db.get(User, user_id) if user_id is not None else None
	if user and user.role == "admin":
		demo_names = {entry["username"] for entry in auth._DEMO_USERS.values()}
		if user.username not in demo_names or (task.user and task.user.username in demo_names):
			return task
	raise HTTPException(status_code=403, detail={"status_code": 403, "error": "forbidden", "message": "无权访问该任务"})


def _task_detail(task):
	return {
		"id": task.id,
		"user_id": task.user_id,
		"patient_id": task.patient_id,
		"patient_age": task.patient_age,
		"patient_symptoms": task.patient_symptoms,
		"patient_duration": task.patient_duration,
		"pipeline_mode": task.pipeline_mode,
		"features": task_features(task),
		"mask_path": task.mask_path,
		"roi_path": task.roi_path,
		"gradcam_path": task.gradcam_path,
		"llm_report": task.llm_report,
		"edited_report": task.edited_report,
		"effective_report": task.edited_report if task.edited_report is not None else task.llm_report,
		"report_edited_at": task.report_edited_at.isoformat() + "Z" if task.report_edited_at else None,
		**extra_fields(task),
		"status": task.status,
		"inference_time_ms": task.inference_time_ms,
		"request_id": task.request_id,
		"created_at": task.created_at.isoformat() + "Z" if task.created_at else None,
		"updated_at": task.updated_at.isoformat() + "Z" if task.updated_at else None,
	}


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: int, user_id: Optional[int] = Depends(get_current_user_id), db: Session = Depends(get_db)):
	return _task_detail(_authorized_task(db, task_id, user_id))


@app.get("/api/v1/tasks/{task_id}/original")
def original_image(task_id: int, user_id: Optional[int] = Depends(get_current_user_id), db: Session = Depends(get_db)):
	task = _authorized_task(db, task_id, user_id, require_identity=True)
	if not task.original_media_type or task.original_image is None:
		raise HTTPException(status_code=404, detail="该任务未保存原图")
	return Response(task.original_image, media_type=task.original_media_type,
					headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@app.patch("/api/v1/tasks/{task_id}/report")
def edit_report(task_id: int, body: ReportRevision, user_id: Optional[int] = Depends(get_current_user_id),
				db: Session = Depends(get_db)):
	task = _authorized_task(db, task_id, user_id, require_identity=True)
	now = datetime.utcnow()
	saved = db.execute(update(AnalysisTask).where(
		AnalysisTask.id == task_id, AnalysisTask.report_version == body.version
	).values(edited_report=body.text, report_version=body.version + 1, report_edited_at=now, updated_at=now))
	if saved.rowcount != 1:
		db.rollback()
		raise HTTPException(status_code=409, detail="报告已在其他窗口保存，请重新载入后核对修订")
	db.commit()
	db.refresh(task)
	return _task_detail(task)


@app.post("/api/v1/tasks/{task_id}/report/retry")
async def retry_report(task_id: int, user_id: Optional[int] = Depends(get_current_user_id), db: Session = Depends(get_db)):
	task = _authorized_task(db, task_id, user_id, require_identity=True)
	retry_statuses = ('failed', 'interrupted')
	if task.status != 'completed' or task.report_status not in retry_statuses:
		raise HTTPException(status_code=409, detail="仅可重试已完成分割且报告失败或中断的任务")
	variant = (task.report_metadata_json or {}).get('variant', 'constrained')
	if variant not in ('text_only', 'constrained', 'image_text'):
		variant = 'constrained'
	result = {'features': task_features(task), 'mode': task.pipeline_mode, 'model_info': task.model_info_json or {}}
	if llm_gateway and variant != 'text_only':
		path = Path(TEMP_DIR) / Path(task.roi_path).name if task.roi_path else None
		if not path or not path.is_file() or (roi := read_image(path)) is None:
			raise HTTPException(status_code=409, detail="该任务的报告输入图像已缺失")
		result['preprocessed'] = {'resized': roi}
	result['segmentation'] = (task.report_metadata_json or {}).get('facts', {}).get('segmentation')
	claimed = db.execute(update(AnalysisTask).where(
		AnalysisTask.id == task_id, AnalysisTask.report_status.in_(retry_statuses)
	).values(report_status='processing'))
	if claimed.rowcount != 1:
		db.rollback()
		raise HTTPException(status_code=409, detail="报告状态已更新，请重新载入任务")
	factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
	# Load report inputs before releasing the request session's transaction.
	lines = _build_markdown_report(result['features'], task.patient_age, task.patient_symptoms, task.patient_duration)
	db.expunge(task)
	db.commit()
	async with aclosing(report_events(factory, task, result, llm_gateway, variant, lines)) as report:
		async for _ in report:
			pass
	return _task_detail(db.get(AnalysisTask, task_id))


@app.delete("/api/v1/tasks/{task_id}")
def delete_task(task_id: int, user_id: Optional[int] = Depends(get_current_user_id), db: Session = Depends(get_db)):
	task = _authorized_task(db, task_id, user_id)
	if task.user_id != user_id:
		raise HTTPException(status_code=403, detail="无权删除该任务")

	for path in _task_artifact_paths(task):
		if path:
			abs_path = os.path.join(TEMP_DIR, path.split("/")[-1])
			if os.path.isfile(abs_path):
				try:
					os.remove(abs_path)
				except OSError:
					pass

	db.delete(task)
	db.commit()
	return {"message": "任务已删除"}


@app.get("/api/v1/tasks/{task_id}/export")
def export_task(
	task_id: int,
	format: str = Query("markdown", pattern="^(markdown|pdf)$"),
	user_id: Optional[int] = Depends(get_current_user_id),
	db: Session = Depends(get_db),
):
	task = _authorized_task(db, task_id, user_id)

	features = task_features(task)
	created_str = task.created_at.strftime("%Y-%m-%d %H:%M:%S") if task.created_at else "N/A"
	report_body = task.edited_report or task.llm_report or "> 来源：本地规则模板（此任务未保存生成报告）。\n\n" + "".join(_build_markdown_report(features, task.patient_age, task.patient_symptoms, task.patient_duration))
	review_status = f"医生修订稿 · 版本 {task.report_version}" if task.edited_report is not None else "系统生成"
	classification = task.classification_json or {"status": "not_evaluated"}
	reliability = task.reliability_json or {"status": "not_evaluated"}
	provenance = json.dumps({"model_info": task.model_info_json, "classification": classification,
	                        "reliability": reliability, "report_status": task.report_status,
	                        "report_metadata": {k: v for k, v in (task.report_metadata_json or {}).items() if k != "facts"}}, ensure_ascii=False, indent=2)
	md_content = f"""# 智肤 SkinVision 辅助分析报告

**任务 ID**：{task.id}  
**分析时间**：{created_str}
**患者信息**：{str(task.patient_age) + " 岁" if task.patient_age is not None else "年龄未提供"}，主诉症状：{task.patient_symptoms}，病程：{task.patient_duration}

## 量化特征

| 指标 | 数值 | 解释或既有规则 |
|------|------|--------|
| 病灶面积占比 | {_metric_text(features, 'area_ratio', 2, '%')} | 分割面积占图像比例，不表示疾病风险 |
| 边缘不规则指数 | {_metric_text(features, 'irregularity_index', 4)} | 既有不规则度阈值，未经临床校准 |
| 色彩变异度 | {_metric_text(features, 'color_variance', 6)} | 区域像素颜色差异 |

既有规则提示：{extra_fields(task)['risk_level']}（{RISK_SOURCE}）

## 模块状态

- 分割执行方式：{task.pipeline_mode or '未记录'}
- 七分类状态：{classification.get('status')}；候选：{', '.join(classification.get('candidates', [])) or '无'}
- 可靠性状态：{reliability.get('status')}（模型预测变化，不表示疾病危险程度）
- 报告状态：{task.report_status or 'not_evaluated'}
- 导出正文：{review_status}

## 辅助分析报告

{report_body}

## 来源与复核信息

```json
{provenance}
```

---
*本报告仅供临床参考，不替代专科医生诊断意见。*
"""
	if format == "markdown":
		return Response(content=md_content, media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="report_{task_id}.md"'})

	try:
		from backend.skinvision.report_export import export_pdf
		pdf_bytes = export_pdf(md_content)
		return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="report_{task_id}.pdf"'})
	except ImportError:
		raise HTTPException(status_code=501, detail="PDF 导出依赖未安装，请按运行依赖清单安装")
	except RuntimeError as exc:
		raise HTTPException(status_code=501, detail=str(exc))



@app.get("/api/v1/admin/stats")
def admin_stats(
	_: None = Depends(require_admin),
	db: Session = Depends(get_db),
):
	total_tasks = db.query(AnalysisTask).count()
	completed_tasks = db.query(AnalysisTask).filter(AnalysisTask.status == "completed").count()
	failed_tasks = db.query(AnalysisTask).filter(AnalysisTask.status == "failed").count()
	processing_tasks = db.query(AnalysisTask).filter(AnalysisTask.status == "processing").count()
	total_users = db.query(User).count()
	active_users = db.query(User).filter(User.is_active.is_(True)).count()
	avg_ms = db.query(func.avg(AnalysisTask.inference_time_ms)).filter(AnalysisTask.inference_time_ms.isnot(None)).scalar()
	return {
		"total_tasks": total_tasks,
		"completed_tasks": completed_tasks,
		"failed_tasks": failed_tasks,
		"processing_tasks": processing_tasks,
		"total_users": total_users,
		"active_users": active_users,
		"avg_inference_ms": round(float(avg_ms), 1) if avg_ms else 0.0,
		"pipeline_mode": pipeline.mode,
	}


@app.get("/api/v1/admin/users")
def admin_users(
	page: int = Query(1, ge=1),
	page_size: int = Query(20, ge=1, le=100),
	_: None = Depends(require_admin),
	db: Session = Depends(get_db),
):
	query = db.query(User).order_by(User.created_at.desc())
	total = query.count()
	users = query.offset((page - 1) * page_size).limit(page_size).all()
	return {
		"data": [
			{
				"id": user.id,
				"username": user.username,
				"email": user.email,
				"role": user.role or "doctor",
				"is_active": user.is_active,
				"last_login": user.last_login.isoformat() + "Z" if user.last_login else None,
				"created_at": user.created_at.isoformat() + "Z",
			}
			for user in users
		],
		"pagination": {
			"page": page,
			"page_size": page_size,
			"total_count": total,
			"total_pages": math.ceil(total / page_size) if page_size else 1,
		},
	}


@app.patch("/api/v1/admin/users/{uid}/status")
def admin_toggle_user_status(
	uid: int,
	_: None = Depends(require_admin),
	db: Session = Depends(get_db),
):
	user = db.query(User).filter(User.id == uid).first()
	if not user:
		raise HTTPException(status_code=404, detail="用户不存在")
	user.is_active = not user.is_active
	user.updated_at = datetime.utcnow()
	db.commit()
	return {"id": user.id, "is_active": user.is_active}


@app.patch("/api/v1/admin/users/{uid}/role")
def admin_update_user_role(
	uid: int,
	body: dict,
	_: None = Depends(require_admin),
	db: Session = Depends(get_db),
):
	role = body.get("role", "doctor")
	if role not in ("doctor", "admin"):
		raise HTTPException(status_code=422, detail="role 只能为 doctor 或 admin")
	user = db.query(User).filter(User.id == uid).first()
	if not user:
		raise HTTPException(status_code=404, detail="用户不存在")
	user.role = role
	user.updated_at = datetime.utcnow()
	db.commit()
	return {"id": user.id, "role": user.role}


@app.get("/api/v1/admin/logs")
def admin_logs(
	page: int = Query(1, ge=1),
	page_size: int = Query(20, ge=1, le=100),
	action: Optional[str] = Query(None),
	_: None = Depends(require_admin),
	db: Session = Depends(get_db),
):
	query = db.query(AuditLog, User.username).outerjoin(User, AuditLog.user_id == User.id)
	if action:
		query = query.filter(AuditLog.action == action)
	query = query.order_by(AuditLog.created_at.desc())
	total = query.count()
	rows = query.offset((page - 1) * page_size).limit(page_size).all()
	items = []
	for log, username in rows:
		detail_username = None
		if isinstance(log.detail_json, dict):
			detail_username = log.detail_json.get("username")
		items.append(
			{
				"id": log.id,
				"user_id": log.user_id,
				"username": username or detail_username or "系统",
				"action": log.action or "unknown",
				"ip_address": log.ip_address,
				"created_at": log.created_at.isoformat() + "Z" if log.created_at else None,
			}
		)
	return {
		"data": items,
		"pagination": {
			"page": page,
			"page_size": page_size,
			"total_count": total,
			"total_pages": math.ceil(total / page_size) if page_size else 1,
		},
	}


@app.get("/api/v1/admin/tasks")
def admin_tasks(
	page: int = Query(1, ge=1),
	page_size: int = Query(15, ge=1, le=100),
	status: Optional[str] = Query(None),
	_: None = Depends(require_admin),
	db: Session = Depends(get_db),
):
	query = db.query(AnalysisTask, User.username).outerjoin(User, AnalysisTask.user_id == User.id)
	if status:
		query = query.filter(AnalysisTask.status == status)
	query = query.order_by(AnalysisTask.created_at.desc())
	total = query.count()
	rows = query.offset((page - 1) * page_size).limit(page_size).all()
	return {
		"data": [
			{
				"id": task.id,
				"user_id": task.user_id,
				"username": username or "未登录用户",
				"patient_id": task.patient_id,
				"patient_age": task.patient_age,
				"irregularity_index": task.irregularity_index,
				"area_ratio": task.area_ratio,
				"risk_level": extra_fields(task)["risk_level"],
				"risk_source": RISK_SOURCE,
				"report_status": task.report_status,
				"status": task.status,
				"pipeline_mode": task.pipeline_mode,
				"inference_time_ms": task.inference_time_ms,
				"created_at": task.created_at.isoformat() + "Z" if task.created_at else None,
			}
			for task, username in rows
		],
		"pagination": {
			"page": page,
			"page_size": page_size,
			"total_count": total,
			"total_pages": math.ceil(total / page_size) if page_size else 1,
		},
	}

# Register last so API routes keep their original behavior.
from backend.skinvision.web import mount_frontend
mount_frontend(app, resolve_path("frontend/dist"))
