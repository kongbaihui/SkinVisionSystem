#!/usr/bin/env python3
"""
API 在线回归脚本 — http://localhost:8000
检查 SkinVision Gateway 的认证、分析和历史接口：
  /health
  /api/v1/auth/register
  /api/v1/auth/login
  /api/v1/auth/refresh
  /api/v1/auth/me
  /api/v1/auth/logout
  /api/v1/diagnostic/analyze
  /api/v1/diagnostic/analyze_stream  (SSE)
    /api/v1/tasks
    /api/v1/tasks/{id}
    /api/v1/tasks/{id}/export
    DELETE /api/v1/tasks/{id}
"""

import json
import os
import sys
import time
import uuid

import httpx

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# 使用一张真实的皮肤图像（也可换成任意 JPEG）
_DEFAULT_IMAGE = os.path.join(
    os.path.dirname(__file__),
    "data/research/isic2016/ISBI2016_ISIC_Part1_Test_Data/ISIC_0000003.jpg",
)
IMAGE_PATH = os.getenv("TEST_IMAGE_PATH", _DEFAULT_IMAGE)

# ── 颜色输出 ────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

_pass_count = 0
_fail_count = 0


def ok(msg: str) -> None:
    global _pass_count
    _pass_count += 1
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")


def fail(msg: str) -> None:
    global _fail_count
    _fail_count += 1
    print(f"  {RED}✗ FAIL{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {YELLOW}{title}{RESET}")
    print(f"{'─' * 60}")


def check(cond: bool, msg: str) -> bool:  # noqa: FBT001
    if cond:
        ok(msg)
    else:
        fail(msg)
    return cond


# ── 共享状态（跨测试传递凭证 / ID）───────────────────────────────────────
state: dict = {
    "access_token": None,
    "refresh_token": None,
    "task_id": None,
}

# 生成独立用户名，避免与已存在的用户冲突
_SUFFIX = uuid.uuid4().hex[:6]
TEST_USER = f"testuser_{_SUFFIX}"
TEST_PASS = "Test1234"
TEST_EMAIL = f"{TEST_USER}@example.com"


def _headers() -> dict:
    token = state["access_token"]
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _json(r: httpx.Response) -> dict:
    """Parse JSON safely; return empty dict on failure."""
    try:
        return r.json()
    except Exception:
        return {}


# ── 测试函数 ─────────────────────────────────────────────────────────────────

def test_health() -> None:
    section("GET /health")
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    check(r.status_code == 200, f"状态码 200（实际 {r.status_code}）")
    body = r.json()
    check("status" in body and body["status"] == "ok", "响应包含 status=ok")
    check("pipeline_mode" in body, "响应包含 pipeline_mode 字段")


def test_register() -> None:
    section("POST /api/v1/auth/register")
    payload = {"username": TEST_USER, "password": TEST_PASS, "email": TEST_EMAIL}
    r = httpx.post(f"{BASE_URL}/api/v1/auth/register", json=payload, timeout=10)
    if not check(r.status_code == 201, f"状态码 201（实际 {r.status_code}）"):
        return
    body = _json(r)
    check("access_token" in body, "响应包含 access_token")
    check("refresh_token" in body, "响应包含 refresh_token")
    check(body.get("username") == TEST_USER, f"username 为 {TEST_USER}")
    state["access_token"] = body.get("access_token")
    state["refresh_token"] = body.get("refresh_token")

    # 重复注册应返回 400
    r2 = httpx.post(f"{BASE_URL}/api/v1/auth/register", json=payload, timeout=10)
    check(r2.status_code == 400, f"重复注册返回 400（实际 {r2.status_code}）")


def test_register_validation() -> None:
    section("POST /api/v1/auth/register — 参数校验")
    # 密码太短
    r = httpx.post(
        f"{BASE_URL}/api/v1/auth/register",
        json={"username": f"v_{_SUFFIX}", "password": "short"},
        timeout=10,
    )
    check(r.status_code == 422, f"密码太短返回 422（实际 {r.status_code}）")

    # 密码无数字
    r2 = httpx.post(
        f"{BASE_URL}/api/v1/auth/register",
        json={"username": f"v2_{_SUFFIX}", "password": "NoDigitsHere"},
        timeout=10,
    )
    check(r2.status_code == 422, f"密码无数字返回 422（实际 {r2.status_code}）")

    # 用户名含非法字符
    r3 = httpx.post(
        f"{BASE_URL}/api/v1/auth/register",
        json={"username": "bad user!", "password": "Test1234"},
        timeout=10,
    )
    check(r3.status_code == 422, f"非法用户名返回 422（实际 {r3.status_code}）")


def test_login() -> None:
    section("POST /api/v1/auth/login")
    r = httpx.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"username": TEST_USER, "password": TEST_PASS},
        timeout=10,
    )
    if not check(r.status_code == 200, f"状态码 200（实际 {r.status_code}）"):
        return
    body = _json(r)
    check("access_token" in body, "响应包含 access_token")
    check("refresh_token" in body, "响应包含 refresh_token")
    check(body.get("token_type") == "Bearer", "token_type=Bearer")
    state["access_token"] = body.get("access_token", state["access_token"])
    state["refresh_token"] = body.get("refresh_token", state["refresh_token"])

    # 密码错误
    r2 = httpx.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"username": TEST_USER, "password": "wrongpassword"},
        timeout=10,
    )
    check(r2.status_code == 401, f"密码错误返回 401（实际 {r2.status_code}）")


def test_auth_me() -> None:
    section("GET /api/v1/auth/me")
    r = httpx.get(f"{BASE_URL}/api/v1/auth/me", headers=_headers(), timeout=10)
    check(r.status_code == 200, f"状态码 200（实际 {r.status_code}）")
    body = _json(r)
    # 认证启用时返回用户对象，未启用时返回提示消息
    if "id" in body:
        check(body.get("username") == TEST_USER, f"username={TEST_USER}")
        check("email" in body, "响应包含 email 字段")
    else:
        check("message" in body, "认证未启用，响应包含 message")


def test_refresh_token() -> None:
    section("POST /api/v1/auth/refresh")
    if not state["refresh_token"]:
        fail("跳过：尚无 refresh_token")
        return
    r = httpx.post(
        f"{BASE_URL}/api/v1/auth/refresh",
        json={"refresh_token": state["refresh_token"]},
        timeout=10,
    )
    if not check(r.status_code == 200, f"状态码 200（实际 {r.status_code}）"):
        return
    body = _json(r)
    check("access_token" in body, "响应包含新 access_token")
    state["access_token"] = body.get("access_token", state["access_token"])

    # 无效 token
    r2 = httpx.post(
        f"{BASE_URL}/api/v1/auth/refresh",
        json={"refresh_token": "invalid.token.here"},
        timeout=10,
    )
    check(r2.status_code in (401, 422), f"无效 token 返回 401/422（实际 {r2.status_code}）")


def test_analyze() -> None:
    section("POST /api/v1/diagnostic/analyze")
    if not os.path.isfile(IMAGE_PATH):
        fail(f"测试图片不存在：{IMAGE_PATH}，跳过")
        return
    with open(IMAGE_PATH, "rb") as f:
        files = {"image": ("test.jpg", f, "image/jpeg")}
        data = {
            "patient_age": "35",
            "symptoms": "皮肤出现不规则色斑",
            "duration": "3个月",
        }
        r = httpx.post(
            f"{BASE_URL}/api/v1/diagnostic/analyze",
            headers=_headers(),
            files=files,
            data=data,
            timeout=60,
        )
    check(r.status_code == 200, f"状态码 200（实际 {r.status_code}）")
    body = _json(r)
    check("id" in body, "响应包含任务 id")
    check("features" in body, "响应包含 features")
    check("mask_path" in body, "响应包含 mask_path")
    check("roi_path" in body, "响应包含 roi_path")
    check(body.get("status") == "completed", f"status=completed（实际 {body.get('status')}）")
    state["task_id"] = body.get("id")
    state["mask_path"] = body.get("mask_path")
    state["roi_path"] = body.get("roi_path")
    print(f"    → task_id={state['task_id']}, pipeline_mode={body.get('pipeline_mode')}")


def test_analyze_validation() -> None:
    section("POST /api/v1/diagnostic/analyze — 参数校验")
    # 缺少 image
    r = httpx.post(
        f"{BASE_URL}/api/v1/diagnostic/analyze",
        headers=_headers(),
        data={"patient_age": "35", "symptoms": "test", "duration": "1天"},
        timeout=10,
    )
    check(r.status_code == 422, f"缺少 image 返回 422（实际 {r.status_code}）")

    # 年龄越界
    if os.path.isfile(IMAGE_PATH):
        with open(IMAGE_PATH, "rb") as f:
            files = {"image": ("test.jpg", f, "image/jpeg")}
            data = {"patient_age": "200", "symptoms": "test", "duration": "1天"}
            r2 = httpx.post(
                f"{BASE_URL}/api/v1/diagnostic/analyze",
                headers=_headers(),
                files=files,
                data=data,
                timeout=10,
            )
        check(r2.status_code == 422, f"年龄越界返回 422（实际 {r2.status_code}）")


def test_analyze_stream() -> None:
    section("POST /api/v1/diagnostic/analyze_stream  (SSE)")
    if not os.path.isfile(IMAGE_PATH):
        fail(f"测试图片不存在：{IMAGE_PATH}，跳过")
        return
    with open(IMAGE_PATH, "rb") as f:
        files = {"image": ("test.jpg", f, "image/jpeg")}
        data = {
            "patient_age": "42",
            "symptoms": "边缘不规则棕色斑块",
            "duration": "6个月",
        }
        with httpx.stream(
            "POST", f"{BASE_URL}/api/v1/diagnostic/analyze_stream",
            headers=_headers(), files=files, data=data, timeout=90,
        ) as response:
            check(response.status_code == 200, f"状态码 200（实际 {response.status_code}）")
            check("text/event-stream" in response.headers.get("Content-Type", ""), "返回事件流")
            events = []
            for line in response.iter_lines():
                if line.startswith("event: "):
                    events.append(line.removeprefix("event: "))
                if events and events[-1] == "done":
                    break
    check("meta" in events, "收到 meta 事件")
    check("done" in events, "收到 done 事件")


def test_get_tasks() -> None:
    section("GET /api/v1/tasks")
    r = httpx.get(
        f"{BASE_URL}/api/v1/tasks",
        headers=_headers(),
        params={"page": 1, "page_size": 5},
        timeout=10,
    )
    check(r.status_code == 200, f"状态码 200（实际 {r.status_code}）")
    body = _json(r)
    check("data" in body, "响应包含 data 列表")
    check("pagination" in body, "响应包含 pagination 对象")
    pagination = body.get("pagination", {})
    check("total_count" in pagination, "pagination 包含 total_count")
    check("total_pages" in pagination, "pagination 包含 total_pages")

    # 过滤 status
    r2 = httpx.get(
        f"{BASE_URL}/api/v1/tasks",
        headers=_headers(),
        params={"status": "completed"},
        timeout=10,
    )
    check(r2.status_code == 200, f"status 过滤返回 200（实际 {r2.status_code}）")

    # 排序
    r3 = httpx.get(
        f"{BASE_URL}/api/v1/tasks",
        headers=_headers(),
        params={"sort": "created_at.asc"},
        timeout=10,
    )
    check(r3.status_code == 200, f"排序参数返回 200（实际 {r3.status_code}）")


def test_get_task_detail() -> None:
    section("GET /api/v1/tasks/{id}")
    task_id = state["task_id"]
    if task_id is None:
        fail("跳过：尚无 task_id（analyze 测试未通过）")
        return
    r = httpx.get(
        f"{BASE_URL}/api/v1/tasks/{task_id}",
        headers=_headers(),
        timeout=10,
    )
    check(r.status_code == 200, f"状态码 200（实际 {r.status_code}）")
    body = _json(r)
    check(body.get("id") == task_id, f"返回正确 task_id={task_id}")
    check("features" in body, "响应包含 features")
    check("llm_report" in body, "响应包含 llm_report 字段")

    # 不存在的任务
    r2 = httpx.get(
        f"{BASE_URL}/api/v1/tasks/99999999",
        headers=_headers(),
        timeout=10,
    )
    check(r2.status_code == 404, f"不存在任务返回 404（实际 {r2.status_code}）")


def test_export_task() -> None:
    section("GET /api/v1/tasks/{id}/export")
    task_id = state["task_id"]
    if task_id is None:
        fail("跳过：尚无 task_id")
        return

    # Markdown 导出
    r = httpx.get(
        f"{BASE_URL}/api/v1/tasks/{task_id}/export",
        headers=_headers(),
        params={"format": "markdown"},
        timeout=10,
    )
    check(r.status_code == 200, f"markdown 导出 200（实际 {r.status_code}）")
    check(
        "markdown" in r.headers.get("Content-Type", "").lower()
        or len(r.content) > 0,
        "markdown 响应有内容",
    )
    check(
        "attachment" in r.headers.get("Content-Disposition", ""),
        "响应头包含 Content-Disposition attachment",
    )

    # 非法 format 参数
    r2 = httpx.get(
        f"{BASE_URL}/api/v1/tasks/{task_id}/export",
        headers=_headers(),
        params={"format": "xlsx"},
        timeout=10,
    )
    check(r2.status_code == 422, f"非法 format 返回 422（实际 {r2.status_code}）")


def test_delete_task() -> None:
    section("DELETE /api/v1/tasks/{id}")
    task_id = state["task_id"]
    if task_id is None:
        fail("跳过：尚无 task_id")
        return
    r = httpx.delete(
        f"{BASE_URL}/api/v1/tasks/{task_id}",
        headers=_headers(),
        timeout=10,
    )
    check(r.status_code == 200, f"状态码 200（实际 {r.status_code}）")
    check("message" in _json(r), "响应包含 message 字段")

    # 再次获取应返回 404
    r2 = httpx.get(
        f"{BASE_URL}/api/v1/tasks/{task_id}",
        headers=_headers(),
        timeout=10,
    )
    check(r2.status_code == 404, f"已删除任务返回 404（实际 {r2.status_code}）")
    state["task_id"] = None


def test_static_outputs() -> None:
    """测试诊断图像的静态文件访问"""
    section("GET /static/outputs/{filename}")
    mask_path = state.get("mask_path")
    roi_path = state.get("roi_path")
    if not mask_path:
        fail("跳过：尚无 mask_path（analyze 测试未通过）")
        return

    r = httpx.get(f"{BASE_URL}{mask_path}", timeout=10)
    check(r.status_code == 200, f"访问 mask 图像 200（实际 {r.status_code}）")
    check(
        r.headers.get("content-type", "").startswith("image/"),
        f"Content-Type image/*（实际 {r.headers.get('content-type')}）",
    )

    if roi_path:
        r2 = httpx.get(f"{BASE_URL}{roi_path}", timeout=10)
        check(r2.status_code == 200, f"访问 roi 图像 200（实际 {r2.status_code}）")

    # 不存在的文件 → 404
    r3 = httpx.get(f"{BASE_URL}/static/outputs/nonexistent_xyz.png", timeout=10)
    check(r3.status_code == 404, f"不存在文件返回 404（实际 {r3.status_code}）")


def test_user_isolation() -> None:
    """注册第二个用户，验证无法访问第一个用户的任务"""
    section("权限隔离：用户 B 不能访问用户 A 的任务")
    suffix_b = uuid.uuid4().hex[:6]
    user_b = f"userb_{suffix_b}"
    r_reg = httpx.post(
        f"{BASE_URL}/api/v1/auth/register",
        json={"username": user_b, "password": "UserB@5678", "email": f"{user_b}@example.com"},
        timeout=10,
    )
    if not check(r_reg.status_code == 201, f"用户 B 注册 201（实际 {r_reg.status_code}）"):
        return
    token_b = r_reg.json().get("access_token", "")

    # 先用用户 A 创建一个任务（若 state["task_id"] 已被删除，重新创建）
    task_id = state.get("last_task_id_for_isolation")
    if not task_id and os.path.isfile(IMAGE_PATH):
        with open(IMAGE_PATH, "rb") as f:
            r_new = httpx.post(
                f"{BASE_URL}/api/v1/diagnostic/analyze",
                headers=_headers(),
                files={"image": ("test.jpg", f, "image/jpeg")},
                data={"patient_age": "30", "symptoms": "测试隔离", "duration": "1天"},
                timeout=60,
            )
        if r_new.status_code == 200:
            task_id = r_new.json().get("id")
            state["last_task_id_for_isolation"] = task_id

    if not task_id:
        fail("跳过：无法获得用户 A 的任务 ID")
        return

    # 用户 B 查看用户 A 的任务 → 403
    r2 = httpx.get(
        f"{BASE_URL}/api/v1/tasks/{task_id}",
        headers={"Authorization": f"Bearer {token_b}"},
        timeout=10,
    )
    check(r2.status_code == 403, f"用户 B 查看 A 的任务 → 403（实际 {r2.status_code}）")

    # 用户 B 删除用户 A 的任务 → 403
    r3 = httpx.delete(
        f"{BASE_URL}/api/v1/tasks/{task_id}",
        headers={"Authorization": f"Bearer {token_b}"},
        timeout=10,
    )
    check(r3.status_code == 403, f"用户 B 删除 A 的任务 → 403（实际 {r3.status_code}）")

    # 用户 B 的任务列表不包含用户 A 的任务
    r4 = httpx.get(
        f"{BASE_URL}/api/v1/tasks",
        headers={"Authorization": f"Bearer {token_b}"},
        timeout=10,
    )
    ids_b = [t["id"] for t in r4.json().get("data", [])]
    check(task_id not in ids_b, f"用户 B 任务列表不含用户 A 的任务 ID={task_id}")

    # 清理：用户 A 删除该任务
    httpx.delete(
        f"{BASE_URL}/api/v1/tasks/{task_id}",
        headers=_headers(),
        timeout=10,
    )


def test_logout() -> None:
    section("POST /api/v1/auth/logout")
    r = httpx.post(
        f"{BASE_URL}/api/v1/auth/logout",
        headers=_headers(),
        json={"refresh_token": state["refresh_token"]},
        timeout=10,
    )
    check(r.status_code == 200, f"状态码 200（实际 {r.status_code}）")
    check("message" in _json(r), "响应包含 message 字段")


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'═' * 60}")
    print(f"  SkinVision API 在线回归")
    print(f"  目标：{BASE_URL}")
    print(f"{'═' * 60}")

    # 连通性检查
    try:
        httpx.get(f"{BASE_URL}/health", timeout=5)
    except httpx.ConnectError:
        print(f"\n{RED}错误：无法连接到 {BASE_URL}，请先启动服务。{RESET}\n")
        sys.exit(1)

    test_health()
    test_register()
    test_register_validation()
    test_login()
    test_auth_me()
    test_refresh_token()
    test_analyze()
    test_analyze_validation()
    test_analyze_stream()
    test_get_tasks()
    test_get_task_detail()
    test_export_task()
    test_static_outputs()
    test_user_isolation()
    test_delete_task()
    test_logout()

    # ── 汇总 ──────────────────────────────────────────────────────────────
    total = _pass_count + _fail_count
    print(f"\n{'═' * 60}")
    print(
        f"  结果：{GREEN}{_pass_count} 通过{RESET} / "
        f"{RED}{_fail_count} 失败{RESET} / 共 {total} 项断言"
    )
    print(f"{'═' * 60}\n")
    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    main()
