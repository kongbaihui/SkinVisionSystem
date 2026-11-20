import os


def get_app_root() -> str:
	return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_path(path_value: str) -> str:
	if not path_value:
		return path_value
	if os.path.isabs(path_value):
		return path_value
	return os.path.join(get_app_root(), path_value)


def resolve_sqlite_url(database_url: str) -> str:
	prefix = "sqlite:///"
	if not database_url.startswith(prefix):
		return database_url
	db_path = database_url[len(prefix):]
	if not db_path or os.path.isabs(db_path):
		return database_url
	normalized = db_path.replace("\\", os.sep)
	if normalized.startswith(f".{os.sep}"):
		normalized = normalized[2:]
	absolute_path = os.path.join(get_app_root(), normalized)
	os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
	return f"{prefix}{absolute_path}"


def get_default_temp_dir() -> str:
	return resolve_path(os.path.join("outputs", "service"))


def load_project_env() -> None:
	try:
		from dotenv import load_dotenv
	except ImportError:
		return
	env_path = os.path.join(get_app_root(), ".env")
	load_dotenv(dotenv_path=env_path)
