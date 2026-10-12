"""Browser contract checks with isolated API fixtures; no live user data is written."""

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    state = {"session_id": "workspace-ui", "context_revision": 1, "stage": "interview",
             "active_run_id": "", "active_interview_id": "", "input_bundle_id": ""}
    bundle = {}
    failures = []
    def route(request):
        nonlocal bundle
        path = request.request.url.split("/api/")[1].split("?")[0]
        payload = request.request.post_data_json or {} if request.request.method in {"POST", "PATCH"} else {}
        result = {}
        if path == "auth/me":
            result = {"user": {"user_id": "ui-user", "account": "ui-account"}}
        elif path == "config":
            result = {"api_version": "2026-08-30.2", "api_capabilities": ["material_batch_delete", "interview_knowledge_batch_import_diagnostics", "project_library_batch_delete", "interview_knowledge_pdf_mode_selection", "project_analysis_background", "resume_version_delete", "resume_session_delete"]}
        elif path == "career/contexts/interview-inputs":
            bundle = {**payload, "session_id": "workspace-bundle", "context_revision": 2}
            result = bundle
        elif path == "career/contexts/workspace-bundle/confirm":
            bundle.update(confirmed=True, context_revision=3)
            result = bundle
        elif path == "career/contexts/workspace-bundle":
            result = bundle
        elif path in {"career/contexts", "career/contexts/workspace-ui"}:
            if request.request.method == "PATCH":
                assert payload["expected_revision"] == state["context_revision"]
                state.update(payload["patch"])
                state["context_revision"] += 1
            result = state
        elif path == "career/interviews":
            result = {"sessions": []}
        elif path == "career/artifacts":
            result = {"records": [], "workspace": {}}
        request.fulfill(status=200, content_type="application/json", body=json.dumps(result))

    output = Path(".tmp/context-validation")
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: failures.append(str(error)))
        page.add_init_script("localStorage.setItem('career-auth-session', JSON.stringify({token:'fixture',user:{user_id:'ui-user'}}));")
        page.route("**/api/**", route)
        page.goto("http://127.0.0.1:8341", wait_until="networkidle")
        page.get_by_label("岗位名称", exact=True).fill("Backend Engineer")
        page.get_by_label("岗位 JD", exact=True).fill("Python APIs and SQL")
        page.get_by_label("简历", exact=True).fill("Python SQL Redis project experience")
        page.get_by_role("button", name="确认并使用这份 JD 与简历").click()
        page.get_by_text("独立导入 · 用户提供", exact=True).wait_for()
        page.reload(wait_until="networkidle")
        page.screenshot(path=str(output / "reload.png"), full_page=True)
        assert page.get_by_label("岗位 JD", exact=True).input_value() == "Python APIs and SQL"
        assert page.get_by_label("简历", exact=True).input_value() == "Python SQL Redis project experience"
        for width, height, name in ((1440, 1100, "desktop"), (390, 844, "mobile")):
            page.set_viewport_size({"width": width, "height": height})
            page.screenshot(path=str(output / (name + ".png")), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), name
        assert not failures, failures
        browser.close()
    print("PASS: import, confirm, refresh restoration, desktop/mobile overflow, browser errors")


if __name__ == "__main__":
    main()
