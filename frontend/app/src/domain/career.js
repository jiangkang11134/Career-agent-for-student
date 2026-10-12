export function recordsOf(records, type) {
  return (records || []).filter((record) => record.record_type === type && record.status !== "deleted");
}

export function recordByRef(records, ref, expectedType) {
  if (!ref?.record_id) return null;
  return (records || []).find((record) => {
    if (expectedType && record.record_type !== expectedType) return false;
    const identityFields = { ResumeRun: "resume_run_id", JDDocument: "jd_id", JDRequirement: "requirement_id", ResumeDocument: "resume_document_id", VerificationReport: "verification_report_id" };
    const field = identityFields[expectedType || record.record_type];
    return Boolean(field) && record[field] === ref.record_id && (ref.revision == null || record.revision === ref.revision);
  }) || null;
}

export function findRun(records, runId) {
  return recordsOf(records, "ResumeRun").find((run) => run.resume_run_id === runId) || null;
}

export function latestRun(records) {
  return recordsOf(records, "ResumeRun").sort(
    (left, right) => Number(right.updated_at || right.created_at || 0) - Number(left.updated_at || left.created_at || 0),
  )[0] || null;
}

export function runContext(records, runId) {
  const run = findRun(records, runId);
  if (!run) return { run: null, jd: null, requirements: [], document: null, report: null };
  const jd = recordByRef(records, run.jd_ref, "JDDocument");
  const requirements = (run.requirement_refs || [])
    .map((ref) => recordsOf(records, "JDRequirement").find((item) => item.requirement_id === ref.record_id))
    .filter(Boolean);
  const document = recordByRef(records, run.resume_document_ref, "ResumeDocument");
  const report = recordByRef(records, run.verification_report_ref, "VerificationReport");
  return { run, jd, requirements, document, report };
}

export function resumeVersions(records, run) {
  if (!run?.jd_ref?.record_id) return [];
  return recordsOf(records, "ResumeDocument")
    .filter((item) => item.jd_ref?.record_id === run.jd_ref.record_id && item.status !== "invalidated")
    .sort((left, right) => Number(right.updated_at || 0) - Number(left.updated_at || 0));
}

export function interviewTitle(session) {
  const mode = session?.mode === "gap_analysis" ? "查漏补缺" : "真实模拟";
  const role = session?.profile?.role_title || session?.role_title || "当前岗位";
  return `${mode} · ${role}`;
}

export function formatTime(value) {
  if (!value) return "时间未知";
  const numeric = Number(value);
  const date = new Date(numeric > 10_000_000_000 ? numeric : numeric * 1000);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
}

export function formatDate(value) {
  if (!value) return "日期未知";
  const numeric = Number(value);
  const date = new Date(numeric > 10_000_000_000 ? numeric : numeric * 1000);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString("zh-CN");
}

export function shortId(value, length = 12) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, length)}...` : text || "-";
}
