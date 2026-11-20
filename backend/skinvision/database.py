import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.skinvision.runtime import resolve_sqlite_url

DATABASE_URL = resolve_sqlite_url(os.getenv("SKINVISION_DB_URL", "sqlite:///./data/skinvision.db"))

_is_sqlite = DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}
_engine_kwargs: dict = {
	"echo": os.getenv("SKINVISION_DB_ECHO", "false").lower() == "true",
	"connect_args": _connect_args,
}
if not _is_sqlite:
	_engine_kwargs["pool_size"] = int(os.getenv("SKINVISION_DB_POOL_SIZE", "10"))
	_engine_kwargs["pool_recycle"] = int(os.getenv("SKINVISION_DB_POOL_RECYCLE", "3600"))

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()