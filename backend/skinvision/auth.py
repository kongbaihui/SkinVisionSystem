import os
from datetime import datetime, timedelta
from typing import Optional

import bcrypt as _bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.skinvision.database import get_db
from backend.skinvision.db_models import User

SECRET_KEY: str = os.getenv("SKINVISION_AUTH_SECRET", "")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = int(os.getenv("SKINVISION_JWT_EXPIRE", "7200"))
REFRESH_TOKEN_EXPIRE_SECONDS = int(os.getenv("SKINVISION_REFRESH_EXPIRE", "604800"))
ENABLE_AUTH: bool = os.getenv("SKINVISION_ENABLE_AUTH", "false").lower() == "true"

bearer_scheme = HTTPBearer(auto_error=False)

_DEMO_TOKEN = "demo_access_token"
_DEMO_ADMIN_TOKEN = "demo_admin_token"
_DEMO_USERS: dict[str, dict] = {
	_DEMO_TOKEN: {"username": "皮肤科医师 - 演示账号", "role": "doctor"},
	_DEMO_ADMIN_TOKEN: {"username": "管理员 - 演示账号", "role": "admin"},
}


def hash_password(password: str) -> str:
	return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
	try:
		return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
	except Exception:
		return False


def _ensure_secret() -> None:
	if not SECRET_KEY or len(SECRET_KEY) < 32:
		raise HTTPException(
			status_code=500,
			detail="SKINVISION_AUTH_SECRET 未配置或长度不足 32 字符",
		)


def create_access_token(user_id: int, username: str) -> str:
	_ensure_secret()
	expire = datetime.utcnow() + timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)
	payload = {
		"sub": str(user_id),
		"username": username,
		"exp": expire,
		"iat": datetime.utcnow(),
		"type": "access",
	}
	return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: int) -> str:
	_ensure_secret()
	expire = datetime.utcnow() + timedelta(seconds=REFRESH_TOKEN_EXPIRE_SECONDS)
	payload = {
		"sub": str(user_id),
		"exp": expire,
		"iat": datetime.utcnow(),
		"type": "refresh",
	}
	return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str, expected_type: str = "access") -> dict:
	_ensure_secret()
	try:
		payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
		if payload.get("type") != expected_type:
			raise JWTError("wrong token type")
		return payload
	except JWTError:
		raise HTTPException(
			status_code=401,
			detail={
				"status_code": 401,
				"error": "unauthorized",
				"message": "无效或过期的令牌",
			},
		)


def get_current_user_id(
	credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
	db: Session = Depends(get_db),
) -> Optional[int]:
	if credentials is None:
		if not ENABLE_AUTH:
			return None
		raise HTTPException(
			status_code=401,
			detail={
				"status_code": 401,
				"error": "unauthorized",
				"message": "缺少 Authorization 头",
			},
		)
	if not ENABLE_AUTH and credentials.credentials in _DEMO_USERS:
		demo = _DEMO_USERS[credentials.credentials]
		lookup = db.query(User).filter_by(username=demo["username"])
		user = lookup.first()
		if user is None:
			# These reserved names cannot be registered or used for password login.
			user = User(**demo, password_hash="!")
			db.add(user)
			try:
				db.commit()
			except IntegrityError:
				db.rollback()
				user = lookup.one()
		return user.id
	payload = decode_token(credentials.credentials, expected_type="access")
	user_id = int(payload["sub"])
	user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
	if not user:
		raise HTTPException(
			status_code=401,
			detail={
				"status_code": 401,
				"error": "unauthorized",
				"message": "用户不存在或已禁用",
			},
		)
	return user_id


def require_admin(
	credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
	db: Session = Depends(get_db),
) -> None:
	if not ENABLE_AUTH:
		return
	if credentials is None:
		raise HTTPException(
			status_code=401,
			detail={"status_code": 401, "error": "unauthorized", "message": "需要登录"},
		)
	token = credentials.credentials
	try:
		payload = decode_token(token, expected_type="access")
		user_id = int(payload["sub"])
	except HTTPException:
		raise HTTPException(
			status_code=401,
			detail={"status_code": 401, "error": "unauthorized", "message": "无效令牌"},
		)
	user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
	if not user or (user.role or "doctor") != "admin":
		raise HTTPException(
			status_code=403,
			detail={"status_code": 403, "error": "forbidden", "message": "需要管理员权限"},
		)
