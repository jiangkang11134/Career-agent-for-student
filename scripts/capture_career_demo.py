from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture reproducible Career UI screenshots.")
    parser.add_argument("--url", default="http://127.0.0.1:8341")
    parser.add_argument("--output", default="docs/competition/screenshots")
    parser.add_argument(
        "--session-reset-only",
        action="store_true",
        help="Run only the deterministic historical-session reset scenario.",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    viewports = {} if args.session_reset_only else {
        "career-desktop": {"width": 1440, "height": 960},
        "career-mobile": {"width": 390, "height": 844},
    }
    with sync_playwright() as playwright:
        browser_candidates = sorted(
            Path.home().glob("AppData/Local/ms-playwright/chromium-*/chrome-*/chrome.exe")
        )
        launch_kwargs = {"headless": True}
        if browser_candidates:
            launch_kwargs["executable_path"] = str(browser_candidates[-1])
        browser = playwright.chromium.launch(**launch_kwargs)
        for name, viewport in viewports.items():
            page = browser.new_page(viewport=viewport, device_scale_factor=1)
            page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
            page.locator("#careerView").wait_for(state="visible", timeout=30_000)
            page.wait_for_timeout(500)
            page.screenshot(path=str(output / f"{name}.png"), full_page=True)
            metrics = page.evaluate(
                """() => ({
                    bodyWidth: document.body.scrollWidth,
                    viewportWidth: window.innerWidth,
                    h1: document.querySelector('h1')?.textContent,
                    stageCount: document.querySelectorAll('.stage').length,
                    resumeFieldCount: ['resumeName', 'resumeHeadline', 'resumeEmail']
                        .filter(id => document.getElementById(id)).length,
                    knowledgeUploadPresent: Boolean(document.getElementById('knowledgeFile')),
                    knowledgeSourceCount: document.querySelectorAll('#knowledgeSources .list-row').length
                })"""
            )
            if (
                not metrics["h1"]
                or metrics["stageCount"] < 5
                or metrics["resumeFieldCount"] != 3
                or not metrics["knowledgeUploadPresent"]
            ):
                raise RuntimeError(f"Career UI did not render correctly: {metrics}")
            if metrics["bodyWidth"] > metrics["viewportWidth"] + 1:
                raise RuntimeError(f"Horizontal page overflow at {name}: {metrics}")
            print(f"{name}: {metrics}")

            page.locator('[data-stage="library"]').click()
            page.wait_for_timeout(100)
            library_metrics = page.evaluate(
                """() => ({
                    active: document.querySelector('[data-view="library"]')?.classList.contains('active'),
                    summaryCount: document.querySelectorAll('#librarySummary .library-stat').length,
                    version: document.getElementById('libraryVersion')?.textContent
                })"""
            )
            if not library_metrics["active"] or library_metrics["summaryCount"] != 4:
                raise RuntimeError(f"Material library is not interactive at {name}: {library_metrics}")

            page.locator(".new-session").click()
            new_session_metrics = page.evaluate(
                """() => ({
                    title: document.getElementById('careerTaskTitle')?.textContent,
                    draft: sessionStorage.getItem('smart-agent-new-session'),
                    activeRun: sessionStorage.getItem('smart-agent-resume-run-id')
                })"""
            )
            if (
                new_session_metrics["title"] != "新求职任务"
                or new_session_metrics["draft"] != "1"
                or new_session_metrics["activeRun"] is not None
            ):
                raise RuntimeError(
                    f"New Career session did not reset task state at {name}: "
                    f"{new_session_metrics}"
                )

            page.locator('[data-stage="interview"]').click()
            page.locator("#gapModeButton").click()
            interview_metrics = page.evaluate(
                """() => ({
                    bodyWidth: document.body.scrollWidth,
                    viewportWidth: window.innerWidth,
                    modeDescription: document.getElementById('interviewModeDescription')?.textContent,
                    questionText: document.getElementById('questionText')?.textContent,
                    knowledgeSourceWidth: document.getElementById('knowledgeSources')?.scrollWidth,
                    workContentWidth: document.querySelector('.work-content')?.clientWidth
                })"""
            )
            if interview_metrics["bodyWidth"] > interview_metrics["viewportWidth"] + 1:
                raise RuntimeError(
                    f"Horizontal interview overflow at {name}: {interview_metrics}"
                )
            if "薄弱知识点" not in (interview_metrics["modeDescription"] or ""):
                raise RuntimeError(
                    f"Gap-analysis mode description is incorrect at {name}: "
                    f"{interview_metrics}"
                )
            source_width = interview_metrics["knowledgeSourceWidth"] or 0
            content_width = interview_metrics["workContentWidth"] or 0
            if source_width > content_width + 1:
                raise RuntimeError(
                    f"Knowledge source list overflow at {name}: {interview_metrics}"
                )
            interview_name = name.replace("career-", "career-interview-")
            page.screenshot(path=str(output / f"{interview_name}.png"), full_page=True)
            print(f"{interview_name}: {interview_metrics}")
            page.close()

        session_page = browser.new_page(viewport={"width": 1280, "height": 800})
        fixture_workspace = {
            "index": {
                "latest_resume_run_id": "resume-run-fixture-4",
                "material_count": 1,
                "usable_fact_count": 1,
                "candidate_fact_counts_by_status": {},
                "material_catalog_version": "a" * 64,
            },
            "recommended_stage": "jd_analysis",
        }
        fixture_records = [
            {
                "record_type": "JDDocument",
                "jd_id": "jd-fixture",
                "role_title": "后端工程师",
                "company": "示例公司",
                "raw_text": "后端工程师 Python",
            },
            {
                "record_type": "JDRequirement",
                "requirement_id": "requirement-fixture",
                "text": "熟悉 Python",
                "category": "hard_skill",
                "required": True,
            },
        ]
        fixture_records.extend(
            {
                "record_type": "ResumeRun",
                "resume_run_id": f"resume-run-fixture-{position}",
                "jd_ref": {"record_id": "jd-fixture", "revision": 1},
                "requirement_refs": [
                    {"record_id": "requirement-fixture", "revision": 1}
                ],
                "material_catalog_version": "a" * 64,
                "status": "created",
                "updated_at": position,
            }
            for position in range(1, 5)
        )

        def fulfill_artifacts(route) -> None:
            route.fulfill(
                json={"workspace": fixture_workspace, "records": fixture_records}
            )

        def delete_fixture_session(route) -> None:
            deleted_id = route.request.url.rsplit("/", 1)[-1]
            for record in fixture_records:
                if record.get("resume_run_id") == deleted_id:
                    record["status"] = "deleted"
            route.fulfill(
                json={
                    "deleted": True,
                    "session_type": "resume",
                    "preserved_requirement_cache_count": 4,
                }
            )

        session_page.route(
            "**/api/career/workspace?goal=resume",
            lambda route: route.fulfill(json=fixture_workspace),
        )
        session_page.route("**/api/career/artifacts", fulfill_artifacts)
        session_page.route(
            "**/api/career/facts?status=candidate",
            lambda route: route.fulfill(json={"facts": []}),
        )
        session_page.route(
            "**/api/career/interviews",
            lambda route: route.fulfill(json={"sessions": []}),
        )
        session_page.route(
            "**/api/career/material-library",
            lambda route: route.fulfill(
                json={
                    "summary": {},
                    "category_counts": {},
                    "material_catalog_version": "a" * 64,
                }
            ),
        )
        session_page.route(
            "**/api/career/resume/templates",
            lambda route: route.fulfill(json={"templates": []}),
        )
        session_page.route("**/api/career/sessions/resume/*", delete_fixture_session)
        session_page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        session_page.locator("#careerView").wait_for(state="visible", timeout=30_000)
        session_page.locator("#recentSessions .session-entry").first.wait_for(
            state="visible", timeout=30_000
        )
        recent_count = session_page.locator("#recentSessions .session-entry").count()
        session_page.locator(".history-link").click()
        history_count = session_page.locator("#historySessions .session-entry").count()
        if recent_count != 3 or history_count != 4:
            raise RuntimeError(
                f"Career session limits are incorrect: recent={recent_count}, "
                f"history={history_count}"
            )
        session_page.locator(
            '#recentSessions .session[data-id="resume-run-fixture-2"]'
        ).click()
        session_page.locator(".new-session").click()
        reset_metrics = session_page.evaluate(
            """() => ({
                summary: document.getElementById('currentJDSummary')?.textContent,
                roleTitle: document.getElementById('jdRoleTitle')?.value,
                company: document.getElementById('jdCompany')?.value,
                jdText: document.getElementById('jdText')?.value,
                requirements: document.getElementById('evidenceList')?.textContent,
                status: document.getElementById('verificationStatus')?.textContent,
                requirementCount: document.getElementById('requirementCountRail')?.textContent
            })"""
        )
        if (
            "尚未添加岗位 JD" not in (reset_metrics["summary"] or "")
            or reset_metrics["roleTitle"]
            or reset_metrics["company"]
            or reset_metrics["jdText"]
            or "提交新的岗位 JD" not in (reset_metrics["requirements"] or "")
            or reset_metrics["status"] != "尚未分析"
            or reset_metrics["requirementCount"] != "0"
        ):
            raise RuntimeError(
                f"New Career session retained historical JD state: {reset_metrics}"
            )
        session_page.screenshot(
            path=str(output / "career-new-session-reset.png"), full_page=True
        )
        print(f"career-new-session-reset: {reset_metrics}")
        session_page.locator(
            '#recentSessions .session[data-id="resume-run-fixture-2"]'
        ).click()
        restored = session_page.evaluate(
            """() => ({
                title: document.getElementById('careerTaskTitle')?.textContent,
                runId: sessionStorage.getItem('smart-agent-resume-run-id'),
                activeType: sessionStorage.getItem('smart-agent-active-session-type'),
                requirementText: document.getElementById('evidenceList')?.textContent
            })"""
        )
        if (
            restored["title"] != "后端工程师"
            or restored["runId"] != "resume-run-fixture-2"
            or restored["activeType"] != "resume"
            or "熟悉 Python" not in (restored["requirementText"] or "")
        ):
            raise RuntimeError(f"Recent Career session could not be restored: {restored}")
        print(f"career-recent-session: {restored}")
        session_page.locator(".history-link").click()
        session_page.once("dialog", lambda dialog: dialog.accept())
        session_page.locator(
            '#historySessions .delete-session[data-id="resume-run-fixture-2"]'
        ).click()
        session_page.wait_for_timeout(200)
        deleted_metrics = session_page.evaluate(
            """() => ({
                historyCount: document.querySelectorAll('#historySessions .session-entry').length,
                deletedVisible: Boolean(document.querySelector(
                    '#historySessions [data-id="resume-run-fixture-2"]'
                )),
                toast: document.getElementById('toast')?.textContent
            })"""
        )
        if (
            deleted_metrics["historyCount"] != 3
            or deleted_metrics["deletedVisible"]
            or "会话已删除" not in (deleted_metrics["toast"] or "")
        ):
            raise RuntimeError(f"Career session deletion failed: {deleted_metrics}")
        print(f"career-session-delete: {deleted_metrics}")
        session_page.close()
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
