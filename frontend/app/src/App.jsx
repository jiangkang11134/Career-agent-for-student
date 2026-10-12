import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity, Archive, Box, Check, ChevronDown, ChevronRight, ChevronUp, FileText, Folder, FolderOpen, History, LayoutDashboard, LoaderCircle,
  MessageCircle, Mic, MicOff, Moon, Pause, Plus, Radio, RefreshCw, Send, Settings2, ShieldCheck, Sun, Trash2, Upload, UserRound,
  X,
} from "lucide-react";
import {
  api, clearAuthSession, getAuthSession, getAuthToken, jsonBody,
  onAuthSessionChange, queryKeys, setAuthSession, uploadBody,
} from "./api/client";
import { formatDate, formatTime, resumeVersions, runContext, shortId } from "./domain/career";
import { useSession } from "./state/session";

const STAGES = [
  ["library", "个人材料库", Archive],
  ["evidence", "岗位 JD", FileText],
  ["resume", "简历", UserRound],
  ["coach", "面试辅导", Radio],
  ["interview", "面试", MessageCircle],
  ["admin", "运行后台", Activity],
];

const THEME_STORAGE_KEY = "smart-agent-theme";
const EXPECTED_CAREER_API_VERSION = "2026-08-30.2";
const MATERIAL_CATEGORY_LABELS = {
  resume: "原始简历",
  education: "教育背景",
  internship: "实习经历",
  research: "科研成果",
  award: "个人奖项",
  skills: "个人技能/其他",
  project: "项目经历",
};
const STRUCTURED_MATERIAL_CONFIG = {
  education: {
    title: "填写教育背景",
    note: "学历、专业、时间、绩点和主修课程会进入候选事实区。",
    fields: [
      ["school", "学校名", "text", true],
      ["major", "专业", "text", true],
      ["degree", "学位层次", "select", false, ["大专", "本科", "硕士", "博士"]],
      ["start_date", "开始时间", "date", false],
      ["end_date", "结束时间", "date", false],
      ["gpa", "绩点", "text", false],
      ["courses", "主修课程", "text", false],
    ],
  },
  internship: {
    title: "填写实习经历",
    note: "公司、部门、岗位、简述、工作内容和时间会作为结构化材料保存。",
    fields: [
      ["company", "实习公司名称", "text", true],
      ["department", "项目部门", "text", false],
      ["role", "实习岗位", "text", true],
      ["summary", "实习简述", "textarea", true],
      ["content", "实习内容", "textarea", false],
      ["start_date", "开始时间", "date", true],
      ["end_date", "结束时间", "date", true],
    ],
  },
  skills: {
    title: "填写个人技能/其他",
    note: "技能文本会按原始段落保存，不会被强行拆散。",
    fields: [["content", "个人技能/其他", "textarea", true]],
  },
  research: {
    title: "填写科研成果",
    note: "论文、课题、实验和研究成果会进入候选事实区。",
    fields: [["content", "科研成果", "textarea", true]],
  },
  award: {
    title: "填写个人奖项",
    note: "竞赛、证书、奖学金和荣誉会进入候选事实区。",
    fields: [["content", "个人奖项", "textarea", true]],
  },
};
const STRUCTURED_CATEGORY_OPTIONS = Object.keys(STRUCTURED_MATERIAL_CONFIG);
const UPLOAD_CATEGORY_OPTIONS = [["resume", "原始简历文件"], ["project", "项目经历"], ["interview_knowledge", "八股资料"]];
const UPLOAD_ACCEPTS = {
  resume: ".txt,.md,.pdf,.docx",
  project: ".zip",
  interview_knowledge: ".txt,.md,.markdown,.pdf,.docx,.ipynb,.png,.jpg,.jpeg,.bmp,.webp,.tif,.tiff,.zip",
};
const PROJECT_ARCHIVE_EXTENSIONS = new Set([".zip"]);
const INTERVIEW_KNOWLEDGE_EXTENSIONS = new Set([
  ".txt",
  ".md",
  ".markdown",
  ".pdf",
  ".docx",
  ".ipynb",
  ".png",
  ".jpg",
  ".jpeg",
  ".bmp",
  ".webp",
  ".tif",
  ".tiff",
]);

const INTERVIEW_MODES = [
  { value: "realistic_simulation", label: "真实模拟", description: "按岗位面试流程进行完整模拟" },
  { value: "gap_analysis", label: "查漏补缺", description: "优先追问岗位要求和薄弱知识点" },
];
const COACH_TYPE_LABELS = {
  project: "个人项目",
  interview_knowledge: "八股知识",
  mixed: "混合问题",
};

const AUTO_FINALIZE_SILENCE_MS = 4000;
const AUDIO_SILENCE_RMS_THRESHOLD = 0.02;
const ADMIN_AGENTS = [
  { id: "smart", name: "Smart Agent / Root Supervisor" },
  { id: "code", name: "Code Agent" },
  { id: "career", name: "Career Agent" },
  { id: "project_analyst", name: "Project Analyst Agent" },
  { id: "interviewer", name: "Interviewer Agent" },
  { id: "application", name: "Application Agent" },
  { id: "resume_verification", name: "Resume Verification Agent" },
  { id: "resume_generator", name: "Resume Generator Agent" },
];

function interviewMode(session) {
  return session?.mode === "gap_analysis" ? "gap_analysis" : "realistic_simulation";
}

function latestActiveInterviewId(sessions = []) {
  return [...sessions]
    .filter((item) => item?.status === "active")
    .sort((a, b) => Number(a?.updated_at || 0) - Number(b?.updated_at || 0))
    .at(-1)?.interview_id || "";
}

function interviewDisplayStatus(session, sessions = []) {
  if (session?.status === "cancelled") return "已取消";
  if (session?.status === "active" && session?.interview_id === latestActiveInterviewId(sessions)) return "进行中";
  if (session?.status === "created") return "待开始";
  return "已结束";
}

function initialTheme() {
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function emptyStructuredFields(category) {
  return Object.fromEntries((STRUCTURED_MATERIAL_CONFIG[category]?.fields || []).map(([key]) => [key, ""]));
}

function emptySkillGroup() {
  return { title: "", content: "" };
}

function serializeSkillGroups(groups = []) {
  return groups
    .map((group) => ({
      title: String(group?.title || "").trim(),
      content: String(group?.content || "").trim(),
    }))
    .filter((group) => group.title || group.content)
    .map((group) => (group.title ? group.title + "：\n" : "") + group.content)
    .join("\n\n");
}

function parseSkillGroups(text) {
  const normalized = String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  if (!normalized) return [emptySkillGroup()];
  const groups = [];
  const paragraphs = normalized.split(/\n\s*\n/).map((item) => item.trim()).filter(Boolean);
  paragraphs.forEach((paragraph) => {
    const lines = paragraph.split("\n").map((line) => line.trim()).filter(Boolean);
    if (!lines.length) return;
    const first = lines[0].match(/^([^:：]{1,64})\s*[:：]\s*(.*)$/);
    if (first && first[1].trim()) {
      groups.push({ title: first[1].trim(), content: [first[2].trim(), ...lines.slice(1)].filter(Boolean).join("\n") });
      return;
    }
    if (lines.length > 1 && lines[0].length <= 64) {
      groups.push({ title: lines[0], content: lines.slice(1).join("\n") });
      return;
    }
    groups.push({ title: "", content: lines.join("\n") });
  });
  return groups.length ? groups : [emptySkillGroup()];
}

function normalizeDateInputValue(value, useMonthEnd = false) {
  const text = String(value || "").trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return text;
  const month = text.match(/^(\d{4})-(\d{2})$/);
  if (!month) return text;
  if (!useMonthEnd) return month[1] + "-" + month[2] + "-01";
  const lastDay = new Date(Number(month[1]), Number(month[2]), 0).getDate();
  return month[1] + "-" + month[2] + "-" + String(lastDay).padStart(2, "0");
}

function parseStructuredMaterialText(category, text) {
  const config = STRUCTURED_MATERIAL_CONFIG[category];
  if (!config) return {};
  const fields = emptyStructuredFields(category);
  if (category === "skills") {
    fields.content = text || "";
    return fields;
  }
  const labelMap = new Map(config.fields.map(([key, label]) => [label, key]));
  const fieldTypes = new Map(config.fields.map(([key, , fieldType]) => [key, fieldType]));
  String(text || "").split(/\r?\n/).forEach((line) => {
    const match = line.match(/^\s*([^:：]+)\s*[:：]\s*(.*)$/);
    if (!match) return;
    const key = labelMap.get(match[1].trim());
    if (key) fields[key] = fieldTypes.get(key) === "date" ? normalizeDateInputValue(match[2], key === "end_date") : match[2] || "";
  });
  if (category === "internship" && !fields.start_date && !fields.end_date) {
    const dateValue = String(text || "").match(/实习日期\s*[:：]\s*([^\r\n]+)/)?.[1] || "";
    const parts = dateValue.split(/(?:\s+(?:至|到|-|~|—|–)\s+|至|到|~|—|–)/).map((item) => item.trim()).filter(Boolean);
    if (parts[0]) fields.start_date = normalizeDateInputValue(parts[0], false);
    if (parts[1]) fields.end_date = normalizeDateInputValue(parts[1], true);
  }
  if (!fields.content && config.fields.length === 1) {
    fields.content = String(text || "").replace(/^\s*[^:：]+\s*[:：]\s*/, "");
  }
  return fields;
}

function useCareerQuery(key, path, enabled = true, options = {}) {
  return useQuery({ queryKey: key, queryFn: () => api(path), enabled, ...options });
}

function careerRecordKey(record) {
  if (!record?.record_type) return "";
  const id = record.resume_run_id || record.jd_id || record.requirement_id || record.resume_document_id || record.verification_report_id || record.record_id || "";
  return id ? record.record_type + ":" + id : "";
}

function mergeCareerRecords(baseRecords = [], incomingRecords = []) {
  const merged = [];
  const index = new Map();
  [...baseRecords, ...incomingRecords].filter(Boolean).forEach((record) => {
    const key = careerRecordKey(record);
    if (!key) {
      merged.push(record);
      return;
    }
    if (!index.has(key)) {
      index.set(key, merged.length);
      merged.push(record);
      return;
    }
    merged[index.get(key)] = record;
  });
  return merged;
}

function recordsFromJDAnalyze(data) {
  return [data?.document, ...(data?.requirements || []), data?.resume_run].filter(Boolean);
}

function adminNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value) : 0;
}

function adminDuration(value) {
  const ms = adminNumber(value);
  return ms < 1000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(1) + " s";
}

function adminStatusClass(status) {
  return String(status || "unknown").toLowerCase().replace(/[^a-z_]/g, "_");
}

function inferAdminStatus(eventType) {
  const text = String(eventType || "").toLowerCase();
  if (text.includes("failed") || text.includes("error")) return "failed";
  if (text.includes("timed_out") || text.includes("timeout")) return "timed_out";
  if (text.includes("finished") || text.includes("completed")) return "completed";
  if (text.includes("started") || text.includes("running")) return "running";
  return "event";
}

function normalizeAdminEvent(raw) {
  const event = { ...(raw || {}) };
  const eventType = event.event_type || event.kind || "runtime";
  event.id = String(event.id || event.event_id || ((event.timestamp || event.created_at || Date.now()) + "-" + Math.random()));
  event.agent = event.agent || event.agent_name || event.task_type || event.agent_type || "Agent";
  event.status = event.status || event.state || inferAdminStatus(eventType);
  event.created_at = event.created_at || event.timestamp || event.started_at;
  event.elapsed_ms = event.elapsed_ms ?? event.duration_ms ?? 0;
  event.parent_id = event.parent_id || event.parent_task_id || event.parent_session_id || "";
  event.child_id = event.child_id || event.child_session_id || event.task_id || "";
  event.kind = event.kind || eventType || "runtime";
  event.tool_name = event.tool_name || event.tool || "";
  event.input_summary = event.input_summary ?? event.message ?? "";
  event.output_summary = event.output_summary ?? event.result ?? event.error ?? "";
  return event;
}

function adminAgentType(item) {
  const text = [
    item.agent_type, item.agent, item.agent_name, item.kind, item.tool_name, item.id,
    item.project_id, item.session_id, item.resume_run_id,
  ].join(" ").toLowerCase();
  if (text.includes("interviewer") || item.kind === "interview_question") return "interviewer";
  if (text.includes("project_analyst") || text.includes("project analyst")) return "project_analyst";
  if (text.includes("resume verification") || text.includes("verify_resume") || text.includes("verification")) return "resume_verification";
  if (text.includes("resume generator") || text.includes("resume_generator") || text.includes("generate_resume") || text.includes("plan_resume")) return "resume_generator";
  if (text.includes("application")) return "application";
  if (text.includes("career agent") || item.kind === "career_agent" || item.agent_type === "career") return "career";
  if (text.includes("code agent") || item.kind === "code_agent" || item.agent_type === "code") return "code";
  if (text.includes("smart agent") || text.includes("root supervisor") || text.includes("root_") || item.agent_type === "smart") return "smart";
  return "";
}

function adminAggregateStatus(events) {
  const statuses = events.map((event) => adminStatusClass(event.status));
  if (statuses.some((status) => ["failed", "error"].includes(status))) return "failed";
  if (statuses.some((status) => ["running", "started"].includes(status))) return "running";
  if (statuses.some((status) => ["timeout", "timed_out"].includes(status))) return "timed_out";
  if (events.length) return events.at(-1)?.status || "completed";
  return "unused";
}

function adminCallKey(typeId, item) {
  if (typeId === "interviewer") return item.session_id || item.interview_id || item.id;
  if (typeId === "project_analyst") return item.id || item.project_id;
  if (typeId === "resume_generator" || typeId === "resume_verification") return item.resume_run_id || item.session_id || item.id;
  return item.session_id || item.parent_id || item.child_id || item.id;
}

function adminCallTitle(typeId, key, events) {
  const agent = ADMIN_AGENTS.find((item) => item.id === typeId)?.name || typeId;
  if (typeId === "interviewer") return "完整面试 / " + key;
  if (typeId === "project_analyst") return "项目分析 / " + (events[0]?.project_id || key);
  if (typeId === "career") return "Career 运行 / " + (events[0]?.session_id || key);
  if (typeId === "resume_generator") return "简历生成 / " + (events[0]?.session_id || key);
  if (typeId === "resume_verification") return "简历核验 / " + (events[0]?.session_id || key);
  return agent + " / " + key;
}

function adminCallsFor(events, typeId) {
  const grouped = new Map();
  events.filter((item) => adminAgentType(item) === typeId).forEach((item) => {
    const key = adminCallKey(typeId, item);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(item);
  });
  return [...grouped.entries()].map(([key, rows]) => {
    rows.sort((a, b) => adminNumber(a.created_at) - adminNumber(b.created_at));
    const first = rows[0] || {};
    const last = rows.at(-1) || first;
    const elapsed = rows.reduce((sum, item) => sum + adminNumber(item.elapsed_ms), 0);
    return {
      id: typeId + "::" + key,
      typeId,
      key,
      events: rows,
      created_at: first.created_at,
      updated_at: last.created_at,
      status: adminAggregateStatus(rows),
      elapsed_ms: elapsed,
      title: adminCallTitle(typeId, key, rows),
    };
  }).sort((a, b) => adminNumber(b.updated_at) - adminNumber(a.updated_at));
}

function adminStepsFor(call) {
  if (!call) return [];
  const rows = [{
    id: call.id + "::call",
    index: 0,
    title: "Agent 调用",
    kind: "agent_call",
    status: call.status,
    created_at: call.created_at,
    elapsed_ms: call.elapsed_ms,
    input: { call_id: call.key, event_count: call.events.length },
    output: { status: call.status, elapsed_ms: call.elapsed_ms },
    raw: call,
  }];
  call.events.forEach((event, eventIndex) => {
    rows.push({
      id: call.id + "::event::" + eventIndex,
      index: rows.length,
      title: event.tool_name || event.kind || "运行记录",
      kind: event.kind || "runtime",
      status: event.status,
      created_at: event.created_at,
      elapsed_ms: event.elapsed_ms,
      input: event.input ?? event.input_summary ?? "",
      output: event.output ?? event.output_summary ?? event.error ?? "",
      raw: event,
    });
    (Array.isArray(event.trace) ? event.trace : []).forEach((trace, traceIndex) => {
      rows.push({
        id: call.id + "::event::" + eventIndex + "::trace::" + traceIndex,
        index: rows.length,
        title: trace.tool || trace.tool_name || "工具调用",
        kind: "tool",
        status: trace.ok === false ? "failed" : (trace.status || "completed"),
        created_at: event.created_at,
        elapsed_ms: trace.elapsed_ms || trace.duration_ms || 0,
        input: trace.input ?? trace.input_summary ?? "",
        output: trace.output ?? trace.output_summary ?? "",
        raw: trace,
      });
    });
  });
  return rows;
}

function adminRedactedText(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return String(text || "")
    .replace(/sk-[A-Za-z0-9_-]{12,}/g, "[API key]")
    .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, "Bearer [redacted]")
    .replace(/(api[_-]?key|authorization|token|password|secret)\s*[:=]\s*[^\s,;]+/gi, "$1: [redacted]");
}

function interviewKnowledgeImportSummary(data) {
  const imported = Number(data?.imported_count || 0);
  const failed = Number(data?.failed_count || 0);
  const skipped = Number(data?.skipped_count || 0);
  const chunks = Number(data?.chunk_count || 0);
  const failedItems = Array.isArray(data?.failed) ? data.failed : [];
  const skippedItems = Array.isArray(data?.skipped) ? data.skipped : [];
  const details = failedItems.slice(0, 3).map((item) => `${item.input_name || "未知文件"}：${item.error || item.reason || "未知错误"}`);
  const skippedDetails = skippedItems.slice(0, 2).map((item) => `${item.input_name || "未知文件"}：${item.reason || "已跳过"}`);
  const suffix = [...details, ...skippedDetails].join("；");
  if (failed || skipped) {
    return `八股资料导入完成：成功 ${imported} 份，失败 ${failed} 份，跳过 ${skipped} 份，生成 ${chunks} 个知识块${suffix ? "；" + suffix : ""}`;
  }
  return `已导入 ${imported} 份八股资料，生成 ${chunks} 个知识块`;
}

function interviewKnowledgeImportError(error) {
  const detail = error?.payload?.detail;
  if (detail && typeof detail === "object") {
    const failedItems = Array.isArray(detail.failed) ? detail.failed : [];
    const failedText = failedItems.slice(0, 5).map((item) => `${item.input_name || "未知文件"}：${item.error || "未知错误"}`).join("；");
    const counts = [`成功 ${Number(detail.imported_count || 0)}`, `失败 ${Number(detail.failed_count || failedItems.length || 0)}`, `跳过 ${Number(detail.skipped_count || 0)}`].join("，");
    return `八股资料导入失败：${counts}${failedText ? "；" + failedText : ""}`;
  }
  return `八股资料导入失败：${error?.message || "未知错误"}`;
}

function fileExtension(name) {
  const match = String(name || "").toLowerCase().match(/\.[^.]+$/);
  return match ? match[0] : "";
}

function App() {
  const queryClient = useQueryClient();
  const session = useSession();
  const [auth, setAuth] = useState(getAuthSession);
  const isAuthenticated = Boolean(auth?.token);
  const [toast, setToast] = useState("");
  const [task, setTask] = useState({ state: "idle", text: "" });
  const [search, setSearch] = useState("");
  const [theme, setTheme] = useState(initialTheme);
  const [optimisticRecords, setOptimisticRecords] = useState([]);
  const [agentModelConfigOpen, setAgentModelConfigOpen] = useState(false);
  const [harnessMode, setHarnessMode] = useState(() => window.localStorage.getItem("smart-harness-mode") || "career");

  const [selectedRunIds, setSelectedRunIds] = useState(() => new Set());
  const [deletingRunIds, setDeletingRunIds] = useState(() => new Set());
  const config = useCareerQuery(queryKeys.config, "/api/config");
  const careerReady = isAuthenticated;
  const apiCompatibilityWarning = useMemo(() => {
    if (config.isLoading) return "";
    if (config.isError || !config.data || config.data.error) return "无法确认后端服务版本，删除操作已禁用。请检查并重启 Smart Agent。";
    const capabilities = Array.isArray(config.data.api_capabilities) ? config.data.api_capabilities : [];
    const requiredCapabilities = [
      "material_batch_delete",
      "interview_knowledge_batch_import_diagnostics",
      "project_library_batch_delete",
      "interview_knowledge_pdf_mode_selection",
      "project_analysis_background",
      "resume_version_delete",
      "resume_session_delete",
    ];
    if (config.data.api_version !== EXPECTED_CAREER_API_VERSION || requiredCapabilities.some((capability) => !capabilities.includes(capability))) {
      return "后端服务版本过旧或未重启，部分操作（包括八股资料导入和删除）不可用。请重启 Smart Agent。";
    }
    return "";
  }, [config.data, config.isError, config.isLoading]);
  // The workspace and artifact endpoints scan large Career histories. Do not
  // let them block the material-library first paint; fetch them only in flows
  // that actually render their data.
  const needsArtifacts = ["evidence", "resume", "coach", "interview", "history"].includes(session.stage);
  const workspace = useCareerQuery(queryKeys.workspace, "/api/career/workspace?goal=resume", false);
  const artifacts = useCareerQuery(queryKeys.artifacts, "/api/career/artifacts", careerReady && needsArtifacts);
  const facts = useCareerQuery(queryKeys.facts, "/api/career/facts?status=candidate", careerReady && session.stage === "library");
  const library = useCareerQuery(queryKeys.library, "/api/career/material-library", careerReady && session.stage === "library");
  const templates = useCareerQuery(queryKeys.templates, "/api/career/resume/templates", careerReady && session.stage === "resume");
  const projects = useCareerQuery(queryKeys.projects, "/api/career/project-library", careerReady && (session.stage === "library" || harnessMode === "code"), {
    // Automatic Project Analyst runs can outlive the upload request. Poll the
    // catalog while it is visible so completion and failures appear without a
    // manual refresh.
    refetchInterval: careerReady && (session.stage === "library" || harnessMode === "code") ? 5000 : false,
  });
  const interviews = useCareerQuery(queryKeys.interviews, "/api/career/interviews", careerReady && ["interview", "history"].includes(session.stage));

  const serverRecords = artifacts.data?.records || [];
  const records = useMemo(() => mergeCareerRecords(serverRecords, optimisticRecords), [serverRecords, optimisticRecords]);
  const state = artifacts.data?.workspace || workspace.data || {};
  const selectedRunId = session.isNewSession ? "" : (session.activeRunId || "");
  const context = runContext(records, selectedRunId);
  const currentInterviews = useMemo(() => {
    if (session.inputBundleId) return (interviews.data?.sessions || []).filter((item) => item.jd_id === `input:${session.inputBundleId}:jd`);
    const runId = context.run?.resume_run_id || "";
    if (!runId) return [];
    const jdId = context.jd?.jd_id || context.run?.jd_ref?.record_id || "";
    return (interviews.data?.sessions || []).filter((item) => (
      item.resume_run_id === runId
      && (!jdId || item.jd_id === jdId)
    ));
  }, [interviews.data?.sessions, context.run?.resume_run_id, context.run?.jd_ref?.record_id, context.jd?.jd_id, session.inputBundleId]);
  const activeInterview = currentInterviews.find((item) => (
    item.interview_id === session.activeInterviewId
  ));
  const notify = (message) => {
    setToast(message);
    window.setTimeout(() => setToast(""), 2800);
  };
  const refresh = (keys) => Promise.all(keys.map((key) => queryClient.invalidateQueries({ queryKey: key })));
  const runTask = async (label, operation, invalidate = []) => {
    setTask({ state: "running", text: label });
    try {
      const result = await operation();
      await refresh(invalidate);
      setTask({ state: "done", text: "操作完成" });
      return result;
    } catch (error) {
      setTask({ state: "error", text: error.message || "操作失败" });
      throw error;
    }
  };

  useEffect(() => onAuthSessionChange((next) => {
    setAuth(next);
    queryClient.clear();
    setOptimisticRecords([]);
  }), [queryClient]);

  useEffect(() => {
    if (!auth?.token) return undefined;
    let cancelled = false;
    api("/api/auth/me")
      .then((data) => {
        if (!cancelled && data?.user) setAuthSession({ token: auth.token, user: data.user });
      })
      .catch(() => {
        if (!cancelled) clearAuthSession();
      });
    return () => { cancelled = true; };
  }, [auth?.token]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#171a1f" : "#f6f8fb");
  }, [theme]);

  useEffect(() => {
    if (task.state === "done") {
      const timer = window.setTimeout(() => setTask({ state: "idle", text: "" }), 2200);
      return () => window.clearTimeout(timer);
    }
    return undefined;
  }, [task.state]);

  const recentJDs = useMemo(() => records
    .filter((item) => item.record_type === "ResumeRun" && item.status !== "deleted")
    .map((run) => {
      const jd = runContext(records, run.resume_run_id).jd;
      const roleTitle = jd?.role_title || "未命名岗位 JD";
      const company = jd?.company || "";
      return {
        id: run.resume_run_id,
        type: "resume",
        title: company ? `${roleTitle} · ${company}` : roleTitle,
        time: run.updated_at,
      };
    })
    .sort((a, b) => Number(b.time || 0) - Number(a.time || 0))
    .filter((item) => `${item.title}`.toLowerCase().includes(search.toLowerCase()))
     .slice(0, 40), [records, search]);
  const allRecentSelected = recentJDs.length > 0 && recentJDs.every((entry) => selectedRunIds.has(entry.id));
  const someRecentSelected = recentJDs.some((entry) => selectedRunIds.has(entry.id));
  useEffect(() => {
    const available = new Set(recentJDs.map((entry) => entry.id));
    setSelectedRunIds((current) => new Set([...current].filter((runId) => available.has(runId))));
  }, [recentJDs]);

  const toggleRunSelection = (runId) => setSelectedRunIds((current) => {
    const next = new Set(current);
    if (next.has(runId)) next.delete(runId); else next.add(runId);
    return next;
  });

  const deleteResumeSession = (entry) => {
    const runId = entry?.id;
    if (!runId || deletingRunIds.size) return;
    if (!window.confirm(`确定删除岗位会话“${entry.title}”吗？`)) return;
    setDeletingRunIds(new Set([runId]));
    runTask(
      "正在删除岗位会话...",
      () => api(`/api/career/sessions/resume/${encodeURIComponent(runId)}`, { method: "DELETE" }),
      [queryKeys.artifacts, queryKeys.workspace],
    ).then(() => {
      setOptimisticRecords((current) => current.filter((record) => record.resume_run_id !== runId));
      setSelectedRunIds((current) => { const next = new Set(current); next.delete(runId); return next; });
      if (session.activeRunId === runId) session.newSession();
      notify("岗位会话已删除");
    }).catch(() => {}).finally(() => setDeletingRunIds(new Set()));
  };

  const deleteSelectedResumeSessions = () => {
    const runIds = recentJDs.filter((entry) => selectedRunIds.has(entry.id)).map((entry) => entry.id);
    if (!runIds.length || deletingRunIds.size) return;
    if (!window.confirm(`确定删除选中的 ${runIds.length} 个岗位会话吗？`)) return;
    setDeletingRunIds(new Set(runIds));
    runTask(
      `正在删除 ${runIds.length} 个岗位会话...`,
      async () => {
        const results = await Promise.allSettled(runIds.map((runId) => api(`/api/career/sessions/resume/${encodeURIComponent(runId)}`, { method: "DELETE" })));
        const failed = results.filter((result) => result.status === "rejected");
        if (failed.length) throw new Error(`有 ${failed.length} 个岗位会话删除失败，请刷新后重试。`);
        return results;
      },
      [queryKeys.artifacts, queryKeys.workspace],
    ).then(() => {
      setOptimisticRecords((current) => current.filter((record) => !runIds.includes(record.resume_run_id)));
      setSelectedRunIds((current) => { const next = new Set(current); runIds.forEach((runId) => next.delete(runId)); return next; });
      if (session.activeRunId && runIds.includes(session.activeRunId)) session.newSession();
      notify(`已删除 ${runIds.length} 个岗位会话`);
    }).catch(() => {}).finally(() => setDeletingRunIds(new Set()));
  };

  const selectStage = (stage) => {
    setHarnessMode("career");
    window.localStorage.setItem("smart-harness-mode", "career");
    session.setStage(stage);
    if (!careerReady) return;
    if (stage === "library") queryClient.prefetchQuery({ queryKey: queryKeys.library, queryFn: () => api("/api/career/material-library") });
    if (stage === "resume") queryClient.prefetchQuery({ queryKey: queryKeys.templates, queryFn: () => api("/api/career/resume/templates") });
    if (stage === "library") queryClient.prefetchQuery({ queryKey: queryKeys.projects, queryFn: () => api("/api/career/project-library") });
    if (["interview", "history"].includes(stage)) queryClient.prefetchQuery({ queryKey: queryKeys.interviews, queryFn: () => api("/api/career/interviews") });
    if (stage === "admin") queryClient.prefetchQuery({ queryKey: queryKeys.adminFlow, queryFn: () => api("/api/admin/agent-flow?limit=500") });
  };
  const selectHarnessMode = (mode) => {
    setHarnessMode(mode);
    window.localStorage.setItem("smart-harness-mode", mode);
    if (mode === "career" && session.stage === "admin") session.setStage("library");
  };
  const handleJDAnalyzed = (data) => {
    const nextRecords = recordsFromJDAnalyze(data);
    if (nextRecords.length) {
      setOptimisticRecords((current) => mergeCareerRecords(current, nextRecords));
      queryClient.setQueryData(queryKeys.artifacts, (current) => ({
        ...(current || {}),
        records: mergeCareerRecords(current?.records || [], nextRecords),
      }));
    }
    const runId = data?.resume_run?.resume_run_id || "";
    if (runId) session.selectRun(runId, "evidence");
  };
  const logout = () => {
    api("/api/auth/logout", { method: "POST" }).catch(() => {}).finally(() => {
      clearAuthSession();
      queryClient.clear();
      setOptimisticRecords([]);
      session.newSession();
    });
  };

  if (!isAuthenticated) {
    return <AuthScreen config={config} theme={theme} setTheme={setTheme} />;
  }

  return (
    <div className={`sa-app sa-${harnessMode}-mode`}>
      <header className="sa-topbar">
        <div className="sa-brand"><span className="sa-mark">S</span><span>面向学生个人的求职助手</span></div>
        <div className="sa-domain"><strong>Smart Harness</strong><div className="sa-harness-switch" role="tablist" aria-label="Agent 模式"><button type="button" className={harnessMode === "career" ? "active" : ""} onClick={() => selectHarnessMode("career")}>Career Agent</button><button type="button" className={harnessMode === "code" ? "active" : ""} onClick={() => selectHarnessMode("code")}>Code Agent</button></div><button type="button" className="sa-domain-link" onClick={() => selectStage("admin")}>运行后台</button></div>
        <div className="sa-user-menu"><span>{auth?.user?.display_name || auth?.user?.account || "已登录"}</span><button type="button" onClick={logout}>退出</button></div>
        <button className="sa-icon-button sa-agent-settings-button" type="button" onClick={() => setAgentModelConfigOpen(true)} title="配置 Career Agent 模型" aria-label="配置 Career Agent 模型"><Settings2 size={16} /></button>
        <button className="sa-theme-toggle" type="button" onClick={() => setTheme((value) => value === "dark" ? "light" : "dark")} title={theme === "dark" ? "切换亮色主题" : "切换暗色主题"} aria-label={theme === "dark" ? "切换亮色主题" : "切换暗色主题"}>{theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}<span>{theme === "dark" ? "亮色" : "暗色"}</span></button>
        <div className="sa-connection"><span className={config.isError ? "dot error" : "dot"} />{config.isError ? "离线" : "服务已连接"}<small>{config.data?.model || "模型读取中"}</small></div>
      </header>
      <div className="sa-layout">
        <aside className="sa-rail">
          <button className="sa-new" onClick={() => { session.newSession(); setTask({ state: "idle", text: "" }); }}><Plus size={16} />新建会话</button>
          <div className="sa-rail-group">工作区</div>
          {harnessMode === "career" && STAGES.map(([id, label, Icon]) => <button key={id} className={`sa-nav ${session.stage === id ? "active" : ""}`} onClick={() => selectStage(id)}><Icon size={17} /><span>{label}</span>{id === "interview" && <b>{currentInterviews.length || ""}</b>}</button>)}
          {harnessMode === "code" && <div className="sa-code-rail-note"><FolderOpen size={18} /><span>选择已导入项目后开始代码对话</span></div>}
          <div className="sa-rail-group sa-history-title"><History size={14} />最近岗位</div>
          <input className="sa-search" placeholder="搜索岗位 JD" value={search} onChange={(event) => setSearch(event.target.value)} />
           <div className="sa-recent">{recentJDs.slice(0, 5).map((item) => <div className="sa-recent-row" key={item.id}><input className="sa-checkbox" type="checkbox" checked={selectedRunIds.has(item.id)} onChange={() => toggleRunSelection(item.id)} aria-label={`选择岗位会话 ${item.title}`} /><button type="button" onClick={() => session.selectRun(item.id, "evidence")}><span>{item.title}</span><small>{formatTime(item.time)}</small></button><button type="button" className="sa-icon-button" title="删除岗位会话" aria-label={`删除岗位会话 ${item.title}`} onClick={() => deleteResumeSession(item)} disabled={deletingRunIds.size > 0}><Trash2 size={13} /></button></div>)}{!recentJDs.length && <span className="sa-empty">暂无岗位 JD</span>}</div>
          <button className="sa-history-link" onClick={() => selectStage("history")}>查看全部岗位 JD <ChevronRight size={14} /></button>
        </aside>
        <main className="sa-main">
          {session.error && <p role="alert">{session.error}</p>}
          <div className="sa-main-head"><div><p className="sa-kicker">{harnessMode === "code" ? "SMART HARNESS / CODE" : "SMART HARNESS / CAREER"}</p><h1>{harnessMode === "code" ? "Code Agent" : stageTitle(session.stage, context)}</h1><p className="sa-subtitle">{harnessMode === "code" ? "选择项目文件，与代码 Agent 对话、修改并验证实现" : stageSubtitle(session.stage, context, state)}</p></div><span className={`sa-task ${task.state}`}>{task.state === "running" && <LoaderCircle size={14} className="spin" />}{task.text || "就绪"}</span></div>
          {harnessMode === "career" && apiCompatibilityWarning && <div className="sa-api-warning"><ShieldCheck size={16} /><span>{apiCompatibilityWarning}</span></div>}
          {harnessMode === "code" && <CodeAgentPanel projects={projects.data?.projects || []} projectsLoading={projects.isLoading} notify={notify} setTask={setTask} />}
          {harnessMode === "career" && session.stage === "library" && <LibraryPanel library={library.data} libraryLoading={library.isLoading} libraryError={library.isError} projects={projects.data?.projects || []} facts={facts.data?.facts || []} runTask={runTask} notify={notify} apiCompatible={!apiCompatibilityWarning} />}
          {harnessMode === "career" && session.stage === "evidence" && <EvidencePanel context={context} runTask={runTask} onSuccess={handleJDAnalyzed} notify={notify} />}
          {harnessMode === "career" && session.stage === "resume" && <ResumePanel context={context} templates={templates.data?.templates || []} records={records} runTask={runTask} notify={notify} onNewResume={() => { session.newSession(); setTask({ state: "idle", text: "" }); }} />}
          {harnessMode === "career" && session.stage === "coach" && <InterviewCoachPanel context={context} runTask={runTask} notify={notify} />}
          {harnessMode === "career" && session.stage === "interview" && <><InterviewImportPanel runTask={runTask} notify={notify} onJdImported={handleJDAnalyzed} /><InterviewModePanel context={context} sessions={currentInterviews} activeInterview={activeInterview} runTask={runTask} onSelect={session.selectInterview} notify={notify} /></>}
          {harnessMode === "career" && session.stage === "history" && <HistoryPanel entries={recentJDs} onSelect={(entry) => session.selectRun(entry.id, "evidence")} selectedIds={selectedRunIds} onToggle={toggleRunSelection} onToggleAll={() => setSelectedRunIds(allRecentSelected ? new Set() : new Set(recentJDs.map((entry) => entry.id)))} allSelected={allRecentSelected} someSelected={someRecentSelected} onDelete={deleteResumeSession} onDeleteSelected={deleteSelectedResumeSessions} deleting={deletingRunIds.size > 0} />}
          {harnessMode === "career" && session.stage === "application" && <EmptyPanel icon={<LayoutDashboard size={30} />} title="一键网申" text="根据已核验简历准备申请表单。" />}
          {harnessMode === "career" && session.stage === "admin" && <AdminPanel notify={notify} />}
        </main>
      </div>
      {agentModelConfigOpen && <AgentModelConfigDialog onClose={() => setAgentModelConfigOpen(false)} notify={notify} />}
      {toast && <div className="sa-toast">{toast}</div>}
    </div>
  );
}

function AuthScreen({ config, theme, setTheme }) {
  const queryClient = useQueryClient();
  const [mode, setMode] = useState("login");
  const [account, setAccount] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const isRegister = mode === "register";

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const payload = { account, password };
      if (isRegister) payload.display_name = displayName;
      const data = await api(isRegister ? "/api/auth/register" : "/api/auth/login", {
        method: "POST",
        body: jsonBody(payload),
        auth: false,
      });
      setAuthSession(data);
      queryClient.clear();
    } catch (submitError) {
      setError(submitError.message || "登录失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="sa-auth-page">
      <header className="sa-topbar">
        <div className="sa-brand"><span className="sa-mark">S</span><span>面向学生个人的求职助手</span></div>
        <button className="sa-theme-toggle" type="button" onClick={() => setTheme((value) => value === "dark" ? "light" : "dark")} title={theme === "dark" ? "切换亮色主题" : "切换暗色主题"} aria-label={theme === "dark" ? "切换亮色主题" : "切换暗色主题"}>{theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}<span>{theme === "dark" ? "亮色" : "暗色"}</span></button>
        <div className="sa-connection"><span className={config.isError ? "dot error" : "dot"} />{config.isError ? "离线" : "服务已连接"}</div>
      </header>
      <main className="sa-auth-main">
        <form className="sa-auth-panel" onSubmit={submit}>
          <div className="sa-auth-head">
            <p className="sa-kicker">CAREER WORKSPACE</p>
            <h1>{isRegister ? "创建账号" : "登录账号"}</h1>
            <p>个人资料库、项目库和面试记录会按账号隔离。</p>
          </div>
          <div className="sa-auth-switch">
            <button type="button" className={!isRegister ? "active" : ""} onClick={() => setMode("login")}>登录</button>
            <button type="button" className={isRegister ? "active" : ""} onClick={() => setMode("register")}>注册</button>
          </div>
          {isRegister && <label>姓名<input value={displayName} onChange={(event) => setDisplayName(event.target.value)} autoComplete="name" /></label>}
          <label>账号<input value={account} onChange={(event) => setAccount(event.target.value)} autoComplete="username" minLength={10} maxLength={20} required /></label>
          <label>密码<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={isRegister ? "new-password" : "current-password"} required minLength={8} /></label>
          {error && <div className="sa-auth-error">{error}</div>}
          <button className="sa-button primary sa-auth-submit" type="submit" disabled={loading}>{loading ? "处理中..." : (isRegister ? "注册并进入" : "登录")}</button>
        </form>
      </main>
    </div>
  );
}

function AgentModelConfigDialog({ onClose, notify }) {
  const queryClient = useQueryClient();
  const configuration = useCareerQuery(queryKeys.agentModelConfig, "/api/career/agent-model-config");
  const [roles, setRoles] = useState([]);
  const [mainAgent, setMainAgent] = useState({ base_url: "", api_key: "", api_key_configured: false });
  const [sharedConnection, setSharedConnection] = useState({ base_url: "", api_key: "" });
  const save = useMutation({
    mutationFn: (item) => api("/api/career/agent-model-config", {
      method: "POST",
      body: jsonBody({ roles: [{ role: item.role, provider: item.provider, model: item.model, base_url: item.base_url, api_key: item.api_key }] }),
    }),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.agentModelConfig, data);
      setRoles((data.roles || []).map((item) => ({ ...item, api_key: "" })));
      notify("Agent 配置已保存");
    },
  });
  const saveMainAgent = useMutation({
    mutationFn: () => api("/api/career/agent-model-config/main-agent", {
      method: "POST",
      body: jsonBody({ base_url: mainAgent.base_url, api_key: mainAgent.api_key }),
    }),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.agentModelConfig, data);
      setMainAgent({ ...(data.main_agent || {}), api_key: "" });
      notify("主 Agent 连接已保存");
    },
  });
  const saveSharedConnection = useMutation({
    mutationFn: () => api("/api/career/agent-model-config/shared-connection", {
      method: "POST",
      body: jsonBody(sharedConnection),
    }),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.agentModelConfig, data);
      setRoles((data.roles || []).map((item) => ({ ...item, api_key: "" })));
      setMainAgent({ ...(data.main_agent || {}), api_key: "" });
      setSharedConnection((current) => ({ ...current, api_key: "" }));
      notify("已更新主 Agent 与全部 Career Agent 的连接配置");
    },
  });

  useEffect(() => {
    if (configuration.data?.roles) {
      const configuredRoles = configuration.data.roles;
      const configuredMainAgent = configuration.data.main_agent || {};
      const urls = [...new Set([configuredMainAgent, ...configuredRoles].map((item) => (item.base_url || "").trim()).filter(Boolean))];
      setRoles(configuredRoles.map((item) => ({ ...item, api_key: "" })));
      setMainAgent({ ...configuredMainAgent, api_key: "" });
      setSharedConnection({ base_url: urls.length === 1 ? urls[0] : "", api_key: "" });
    }
  }, [configuration.data]);

  const updateRole = (role, field, value) => {
    setRoles((current) => current.map((item) => item.role === role ? { ...item, [field]: value } : item));
  };

  return <div className="sa-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="sa-modal sa-agent-model-modal" role="dialog" aria-modal="true" aria-labelledby="agent-model-config-title">
      <header className="sa-modal-head"><div><h2 id="agent-model-config-title">Agent 配置</h2><p>主 Agent 与 Career Agent 的连接配置独立保存。</p></div><button type="button" className="sa-icon-button" onClick={onClose} title="关闭" aria-label="关闭"><X size={16} /></button></header>
      {configuration.isLoading && <div className="sa-modal-state"><LoaderCircle className="spin" size={18} />读取配置中</div>}
      {configuration.isError && <div className="sa-modal-state error">{configuration.error?.message || "无法读取 Agent 配置"}</div>}
      {!configuration.isLoading && !configuration.isError && <>
        <div className="sa-shared-connection">
          <div className="sa-shared-connection-head"><strong>统一连接</strong><span>更新主 Agent 与全部 Career Agent</span></div>
          <label>Base URL<input value={sharedConnection.base_url} onChange={(event) => setSharedConnection((current) => ({ ...current, base_url: event.target.value }))} placeholder="https://..." /></label>
          <label>API Key<input type="password" value={sharedConnection.api_key} onChange={(event) => setSharedConnection((current) => ({ ...current, api_key: event.target.value }))} placeholder="请输入 API Key" autoComplete="new-password" /></label>
          <button type="button" className="sa-button primary" onClick={() => saveSharedConnection.mutate()} disabled={!sharedConnection.base_url.trim() || !sharedConnection.api_key.trim() || saveSharedConnection.isPending}>{saveSharedConnection.isPending && <LoaderCircle className="spin" size={14} />}更新全部</button>
          {saveSharedConnection.isError && <p className="sa-shared-connection-error">{saveSharedConnection.error?.message || "批量更新失败"}</p>}
        </div>
        <div className="sa-shared-connection">
          <div className="sa-shared-connection-head"><strong>主 Agent</strong><span>{mainAgent.api_key_configured ? "API Key 已配置" : "需要 API Key"}</span></div>
          <label>Base URL<input value={mainAgent.base_url || ""} onChange={(event) => setMainAgent((current) => ({ ...current, base_url: event.target.value }))} placeholder="https://..." /></label>
          <label>API Key<input type="password" value={mainAgent.api_key || ""} onChange={(event) => setMainAgent((current) => ({ ...current, api_key: event.target.value }))} placeholder={mainAgent.api_key_configured ? "留空不修改" : "请输入 API Key"} autoComplete="new-password" /></label>
          <button type="button" className="sa-button" onClick={() => saveMainAgent.mutate()} disabled={!mainAgent.base_url?.trim() || (!mainAgent.api_key_configured && !mainAgent.api_key?.trim()) || saveMainAgent.isPending}>{saveMainAgent.isPending && <LoaderCircle className="spin" size={14} />}保存</button>
          {saveMainAgent.isError && <p className="sa-shared-connection-error">{saveMainAgent.error?.message || "主 Agent 保存失败"}</p>}
        </div>
        <div className="sa-agent-model-list">
          {roles.map((item) => <div className="sa-agent-model-row" key={item.role}>
          <div className="sa-agent-model-name"><strong>{item.label}</strong><small>{item.api_key_configured ? "API Key 已配置" : "需要 API Key"}</small></div>
          <label>提供商<select value={item.provider} onChange={(event) => updateRole(item.role, "provider", event.target.value)}><option value="custom">Custom</option><option value="openai">OpenAI</option><option value="openrouter">OpenRouter</option><option value="anthropic">Anthropic</option></select></label>
          <label>模型<input value={item.model || ""} onChange={(event) => updateRole(item.role, "model", event.target.value)} placeholder="模型 ID" /></label>
          <label>Base URL<input value={item.base_url || ""} onChange={(event) => updateRole(item.role, "base_url", event.target.value)} placeholder="https://..." /></label>
          <label>API Key<input type="password" value={item.api_key || ""} onChange={(event) => updateRole(item.role, "api_key", event.target.value)} placeholder={item.api_key_configured ? "留空不修改" : "请输入 API Key"} autoComplete="new-password" /></label>
          <button type="button" className="sa-button" onClick={() => save.mutate(item)} disabled={save.isPending}>{save.isPending && <LoaderCircle className="spin" size={14} />}保存</button>
          </div>)}
        </div>
      </>}
      <footer className="sa-modal-actions"><button type="button" className="sa-button ghost" onClick={onClose}>关闭</button></footer>
      {save.isError && <p className="sa-modal-error">{save.error?.message || "保存配置失败"}</p>}
    </section>
  </div>;
}

function stageTitle(stage, context) {
  if (stage === "evidence") return context.jd?.role_title || "岗位 JD";
  if (stage === "resume") return context.jd?.role_title ? `${context.jd.role_title} · 简历` : "简历工作台";
  if (stage === "coach") return context.jd?.role_title ? `${context.jd.role_title} · 面试辅导` : "面试辅导";
  if (stage === "interview") return context.jd?.role_title ? `${context.jd.role_title} · 面试工作台` : "面试工作台";
  if (stage === "admin") return "运行后台";
  return STAGES.find(([id]) => id === stage)?.[1] || "Career";
}

function stageSubtitle(stage, context, state) {
  if (stage === "evidence") return context.jd ? `${context.jd.company || "目标公司"} · ${context.requirements.length} 项岗位要求` : "提交目标岗位 JD，创建独立求职会话";
  if (stage === "resume") return context.run ? `当前版本 ${shortId(context.document?.resume_document_id || context.run.resume_run_id)} · 一键生成、核验并导出` : "确认个人事实后生成证据约束简历";
  if (stage === "coach") return "基于导入项目库、项目 RAG 和 interview-knowledge 八股库回答面试问题";
  if (stage === "interview") return context.run ? "选择真实模拟或查漏补缺，面试只读取当前岗位会话" : "先提交岗位 JD，再选择面试模式";
  if (stage === "admin") return "查看 8 个 Agent 的运行记录、工具调用摘要和实时事件";
  if (stage === "library") return `长期材料独立保存 · ${state.index?.material_count || 0} 份材料 · ${state.index?.candidate_fact_count || 0} 条待确认事实`;
  return "按需加载当前模块数据";
}

function ProjectAnalysisReview({ project, analysis, description, onConfirm }) {
  const understanding = analysis?.understanding || {};
  const list = (value, fallback = []) => {
    if (Array.isArray(value) && value.length) return value.filter(Boolean).map((item) => String(item));
    if (value) return [String(value)];
    return fallback;
  };
  const renderList = (label, value, fallback = []) => {
    const items = list(value, fallback);
    if (!items.length) return null;
    return <div className="sa-project-analysis-field"><strong>{label}</strong><ul>{items.map((item, index) => <li key={label + "-" + index}>{item}</li>)}</ul></div>;
  };
  const sourceRefs = description?.source_refs || understanding.source_refs || [];
  const title = description?.title || understanding.title || project.display_name || project.project_id;
  const goals = description?.goals?.length ? description.goals : list(understanding.goal);
  const coreFlow = description?.core_flow?.length ? description.core_flow : understanding.core_flow;
  const architecture = description?.architecture || understanding.architecture;
  const technologies = description?.technologies?.length ? description.technologies : understanding.technologies;
  const modules = description?.core_modules?.length ? description.core_modules : understanding.core_modules;
  const features = description?.implemented_features?.length ? description.implemented_features : understanding.implemented_features;
  const responsibilities = description?.responsibilities?.length ? description.responsibilities : understanding.candidate_responsibilities;
  const outcomes = description?.outcomes?.length ? description.outcomes : understanding.candidate_outcomes;
  const unknowns = description?.unknowns?.length ? description.unknowns : understanding.unknowns;
  return <div className="sa-project-analysis">
    <div className="sa-project-analysis-head">
      <div><strong>{title}</strong><span>{description?.status === "needs_user_confirmation" ? "待确认个人事实" : "项目分析"}</span></div>
      {description?.status === "needs_user_confirmation" && <button type="button" className="sa-button primary" onClick={() => onConfirm(project, description)}><Check size={15} />确认项目事实</button>}
    </div>
    {!description && <p className="sa-project-analysis-warning">当前分析未形成可确认项目档案，请重新分析。</p>}
    {description?.background && <p className="sa-project-analysis-summary">{description.background}</p>}
    {understanding.analysis_summary && !description?.background && <p className="sa-project-analysis-summary">{understanding.analysis_summary}</p>}
    <div className="sa-project-analysis-grid">
      {renderList("项目目标", goals)}
      {renderList("核心流程", coreFlow)}
      {architecture && <div className="sa-project-analysis-field"><strong>架构</strong><p>{architecture}</p></div>}
      {renderList("技术栈", technologies)}
      {renderList("核心模块", modules)}
      {renderList("已实现功能", features)}
      {renderList("候选职责", responsibilities)}
      {renderList("候选成果", outcomes)}
      {renderList("未知项", unknowns)}
    </div>
    {sourceRefs.length > 0 && <div className="sa-project-analysis-field sa-project-analysis-sources"><strong>源码证据</strong><ul>{sourceRefs.slice(0, 30).map((ref, index) => { const locator = ref.locator || {}; const path = locator.path || ref.path || "未知路径"; const line = locator.line || ref.line || ""; const excerpt = ref.excerpt || ""; return <li key={path + "-" + line + "-" + index}><code>{path}{line ? ":" + line : ""}</code>{excerpt ? " - " + excerpt : ""}</li>; })}</ul></div>}
  </div>;
}

function LibraryPanel({ library, libraryLoading = false, libraryError = false, projects, facts, runTask, notify, apiCompatible = true }) {
  const [category, setCategory] = useState("resume");
  const [files, setFiles] = useState([]);
  const [pdfModes, setPdfModes] = useState({});
  const [selectedMaterialIds, setSelectedMaterialIds] = useState(() => new Set());
  const selectAllMaterialsRef = useRef(null);
  const [selectedProjectIds, setSelectedProjectIds] = useState(() => new Set());
  const selectAllProjectsRef = useRef(null);
  const [selectedKnowledgeIds, setSelectedKnowledgeIds] = useState(() => new Set());
  const selectAllKnowledgeRef = useRef(null);
  const [structuredCategory, setStructuredCategory] = useState("education");
  const [structuredFields, setStructuredFields] = useState(() => emptyStructuredFields("education"));
  const [skillGroups, setSkillGroups] = useState(() => [emptySkillGroup()]);
  const [editingMaterialId, setEditingMaterialId] = useState("");
  const activeStructured = STRUCTURED_MATERIAL_CONFIG[structuredCategory];
  const categoryCounts = library?.category_counts || {};
  const materials = library?.materials || [];
  const knowledgeSources = library?.interview_knowledge_sources || [];
  const projectDescriptions = library?.project_descriptions || [];
  const projectDescriptionsById = useMemo(() => new Map(projectDescriptions.map((item) => [item.project_id, item])), [projectDescriptions]);
  const projectCatalogEntries = useMemo(() => projects
    .map((entry) => {
      const project = entry.project_source || {};
      return {
        project,
        analysis: entry.latest_analysis,
        analysisStatus: entry.analysis_status || (entry.latest_analysis ? "completed" : "pending"),
        latestAnalysisRun: entry.latest_analysis_run || null,
        description: entry.project_description || projectDescriptionsById.get(project.project_id) || null,
      };
    })
    .filter(({ project }) => project.project_id), [projects, projectDescriptionsById]);
  useEffect(() => {
    const available = new Set(materials.map((material) => material.material_id));
    setSelectedMaterialIds((current) => new Set([...current].filter((materialId) => available.has(materialId))));
  }, [library?.material_catalog_version, materials.length]);
  useEffect(() => {
    const available = new Set(projectCatalogEntries.map(({ project }) => project.project_id));
    setSelectedProjectIds((current) => new Set([...current].filter((projectId) => available.has(projectId))));
  }, [library?.material_catalog_version, projectCatalogEntries.length]);
  useEffect(() => {
    const available = new Set(knowledgeSources.map((source) => source.source_id));
    setSelectedKnowledgeIds((current) => new Set([...current].filter((sourceId) => available.has(sourceId))));
  }, [library?.material_catalog_version, knowledgeSources.length]);
  const allMaterialsSelected = materials.length > 0 && selectedMaterialIds.size === materials.length;
  const someMaterialsSelected = selectedMaterialIds.size > 0 && !allMaterialsSelected;
  const allProjectsSelected = projectCatalogEntries.length > 0 && selectedProjectIds.size === projectCatalogEntries.length;
  const someProjectsSelected = selectedProjectIds.size > 0 && !allProjectsSelected;
  const allKnowledgeSelected = knowledgeSources.length > 0 && selectedKnowledgeIds.size === knowledgeSources.length;
  const someKnowledgeSelected = selectedKnowledgeIds.size > 0 && !allKnowledgeSelected;
  useEffect(() => {
    if (selectAllMaterialsRef.current) selectAllMaterialsRef.current.indeterminate = someMaterialsSelected;
  }, [someMaterialsSelected]);
  useEffect(() => {
    if (selectAllProjectsRef.current) selectAllProjectsRef.current.indeterminate = someProjectsSelected;
  }, [someProjectsSelected]);
  useEffect(() => {
    if (selectAllKnowledgeRef.current) selectAllKnowledgeRef.current.indeterminate = someKnowledgeSelected;
  }, [someKnowledgeSelected]);
  if (libraryLoading) {
    return <section className="sa-panel-grid"><div className="sa-card sa-wide sa-loading-state"><LoaderCircle className="spin" size={18} /><strong>正在读取个人材料库...</strong><span>正在加载已保存的材料、项目和待确认事实</span></div></section>;
  }
  if (libraryError) {
    return <section className="sa-panel-grid"><div className="sa-card sa-wide sa-loading-state error"><strong>个人材料库读取失败</strong><span>请刷新页面或检查后端服务状态</span></div></section>;
  }
  const activeStructuredMaterials = materials.filter((material) => material.category === structuredCategory);
  const deleteMaterial = (material) => {
    if (!window.confirm(`确认删除“${material.label || material.original_name}”？关联候选事实和导入副本也会失效。`)) return;
    runTask("正在删除材料...", () => api("/api/career/materials/" + encodeURIComponent(material.material_id), { method: "DELETE" }), [queryKeys.library, queryKeys.facts, queryKeys.workspace, queryKeys.artifacts]).then((data) => {
      notify(`材料已删除${(data.deleted_material_ids || []).length > 1 ? `（含 ${(data.deleted_material_ids || []).length - 1} 个简历分段）` : ""}`);
    }).catch((error) => {
      notify(`材料删除失败：${error.message || "未知错误"}`);
    });
  };
  const deleteProject = (project) => {
    if (!window.confirm(`确认删除项目“${project.display_name || project.project_id}”？项目分析、事实和检索索引会一并清理。`)) return;
    runTask("正在删除项目...", () => api("/api/career/project-library/" + encodeURIComponent(project.project_id) + "?expected_revision=" + encodeURIComponent(project.revision || ""), { method: "DELETE" }), [queryKeys.library, queryKeys.projects, queryKeys.facts, queryKeys.workspace, queryKeys.artifacts]).then(() => {
      notify("项目及其关联数据已删除");
    }).catch((error) => {
      notify(`项目删除失败：${error.message || "未知错误"}`);
    });
  };
  const deleteKnowledge = (source) => {
    if (!apiCompatible) return notify("当前后端版本不兼容，请先重启 Smart Agent");
    if (!window.confirm(`确认删除八股资料“${source.name || source.source_id}”？知识块、向量索引和图索引会一并清理。`)) return;
    runTask("正在删除八股资料...", () => api("/api/career/interview-knowledge/" + encodeURIComponent(source.source_id), { method: "DELETE" }), [queryKeys.library, queryKeys.workspace, queryKeys.artifacts]).then(() => {
      notify("八股资料及其索引已删除");
    }).catch((error) => {
      notify(`八股资料删除失败：${error.message || "未知错误"}`);
    });
  };
  const requestProjectAnalysis = (projectId, background = false) => api(
    "/api/career/project-library/" + encodeURIComponent(projectId) + "/analyze",
    { method: "POST", body: jsonBody({ mode: "agent", intent: "overview", refresh: true, background }) },
  ).then((data) => {
    if (!background && !data?.analysis) throw new Error("项目分析接口未返回分析结果");
    return data;
  });
  const analyzeProject = (project) => runTask(
    "正在分析项目 " + (project.display_name || project.project_id) + "...",
    () => requestProjectAnalysis(project.project_id),
    [queryKeys.library, queryKeys.projects, queryKeys.facts, queryKeys.workspace, queryKeys.artifacts],
  ).then(() => notify("项目分析完成，请确认候选事实"))
    .catch((error) => notify("项目分析失败：" + (error.message || "未知错误")));
  const confirmProjectDescription = (project, description) => {
    if (!description || description.revision == null) {
      notify("项目分析结果缺少修订号，无法确认，请重新分析");
      return;
    }
    runTask(
      "正在确认项目事实...",
      () => api("/api/career/experiences/projects/" + encodeURIComponent(project.project_id) + "/confirm", {
        method: "POST",
        body: jsonBody({ expected_revision: description.revision }),
      }),
      [queryKeys.library, queryKeys.projects, queryKeys.facts, queryKeys.workspace, queryKeys.artifacts],
    ).then(() => notify("项目事实已确认，已进入个人材料库"))
      .catch((error) => notify("项目事实确认失败：" + (error.message || "未知错误")));
  };
  const toggleMaterial = (materialId) => setSelectedMaterialIds((current) => {
    const next = new Set(current);
    if (next.has(materialId)) next.delete(materialId); else next.add(materialId);
    return next;
  });
  const toggleAllMaterials = () => setSelectedMaterialIds(allMaterialsSelected
    ? new Set()
    : new Set(materials.map((material) => material.material_id)));
  const deleteSelectedMaterials = () => {
    if (!apiCompatible) return notify("当前后端版本不兼容，请先重启 Smart Agent");
    const materialIds = [...selectedMaterialIds];
    if (!materialIds.length) return notify("请先勾选要删除的材料");
    if (!window.confirm(`确认删除已勾选的 ${materialIds.length} 份材料？关联候选事实、导入副本和项目索引会按单条删除逻辑一并清理。`)) return;
    runTask(`正在删除 ${materialIds.length} 份材料...`, () => api("/api/career/materials/delete-batch", { method: "POST", body: jsonBody({ material_ids: materialIds }) }), [queryKeys.library, queryKeys.projects, queryKeys.facts, queryKeys.workspace, queryKeys.artifacts]).then((data) => {
      const failed = (data.results || []).filter((item) => !item.deleted);
      setSelectedMaterialIds(new Set());
      if (failed.length) {
        notify(`已删除 ${data.deleted_count || 0} 份材料，${failed.length} 份失败：${failed.map((item) => item.error || item.material_id).join("；")}`);
      } else {
        notify(`已删除 ${data.deleted_count || materialIds.length} 份材料`);
      }
    }).catch((error) => {
      notify(`材料批量删除失败：${error.message || "未知错误"}`);
    });
  };
  const toggleProject = (projectId) => setSelectedProjectIds((current) => {
    const next = new Set(current);
    if (next.has(projectId)) next.delete(projectId); else next.add(projectId);
    return next;
  });
  const toggleAllProjects = () => setSelectedProjectIds(allProjectsSelected
    ? new Set()
    : new Set(projectCatalogEntries.map(({ project }) => project.project_id)));
  const deleteSelectedProjects = () => {
    if (!apiCompatible) return notify("当前后端版本不兼容，请先重启 Smart Agent");
    const selectedProjects = projectCatalogEntries
      .filter(({ project }) => selectedProjectIds.has(project.project_id))
      .map(({ project }) => ({ project_id: project.project_id, expected_revision: project.revision || null }));
    if (!selectedProjects.length) return notify("请先勾选要删除的项目");
    if (!window.confirm(`确认删除已勾选的 ${selectedProjects.length} 个项目？项目库、分析记录、项目事实和相关索引会按单条删除逻辑一并清理。`)) return;
    runTask(`正在删除 ${selectedProjects.length} 个项目...`, () => api("/api/career/project-library/delete-batch", { method: "POST", body: jsonBody({ projects: selectedProjects }) }), [queryKeys.library, queryKeys.projects, queryKeys.facts, queryKeys.workspace, queryKeys.artifacts]).then((data) => {
      const failed = (data.results || []).filter((item) => !item.deleted);
      setSelectedProjectIds(new Set());
      if (failed.length) {
        notify(`已删除 ${data.deleted_count || 0} 个项目，${failed.length} 个失败：${failed.map((item) => item.error || item.project_id).join("；")}`);
      } else {
        notify(`已删除 ${data.deleted_count || selectedProjects.length} 个项目`);
      }
    }).catch((error) => {
      notify(`项目批量删除失败：${error.message || "未知错误"}`);
    });
  };
  const toggleKnowledge = (sourceId) => setSelectedKnowledgeIds((current) => {
    const next = new Set(current);
    if (next.has(sourceId)) next.delete(sourceId); else next.add(sourceId);
    return next;
  });
  const toggleAllKnowledge = () => setSelectedKnowledgeIds(allKnowledgeSelected
    ? new Set()
    : new Set(knowledgeSources.map((source) => source.source_id)));
  const deleteSelectedKnowledge = () => {
    if (!apiCompatible) return notify("当前后端版本不兼容，请先重启 Smart Agent");
    const sourceIds = [...selectedKnowledgeIds];
    if (!sourceIds.length) return notify("请先勾选要删除的八股资料");
    if (!window.confirm(`确认删除已勾选的 ${sourceIds.length} 份八股资料？知识块、向量索引和图索引会一并清理，不会删除电脑上的原始文件。`)) return;
    runTask(`正在删除 ${sourceIds.length} 份八股资料...`, () => api("/api/career/interview-knowledge/delete-batch", { method: "POST", body: jsonBody({ source_ids: sourceIds }) }), [queryKeys.library, queryKeys.workspace, queryKeys.artifacts]).then((data) => {
      const failed = (data.results || []).filter((item) => !item.deleted);
      setSelectedKnowledgeIds(new Set());
      if (failed.length) {
        notify(`已删除 ${data.deleted_count || 0} 份，${failed.length} 份失败：${failed.map((item) => item.error || item.source_id).join("；")}`);
      } else {
        notify(`已删除 ${data.deleted_count || sourceIds.length} 份八股资料`);
      }
    }).catch((error) => {
      notify(`批量删除失败：${error.message || "未知错误"}`);
    });
  };
  const confirmFacts = (decision) => runTask(decision === "confirm" ? "正在确认事实..." : "正在拒绝事实...", () => api("/api/career/facts/decision", { method: "POST", body: jsonBody({ decision, candidate_fact_ids: facts.filter((item) => item.status === "candidate").map((item) => item.candidate_fact_id), reason: "user decision from React Career UI" }) }), [queryKeys.facts, queryKeys.workspace, queryKeys.artifacts]).catch(() => {});
  const chooseStructuredCategory = (nextCategory) => {
    setStructuredCategory(nextCategory);
    setStructuredFields(emptyStructuredFields(nextCategory));
    setSkillGroups(nextCategory === "skills" ? [emptySkillGroup()] : []);
    setEditingMaterialId("");
  };
  const resetStructuredForm = () => {
    setStructuredFields(emptyStructuredFields(structuredCategory));
    if (structuredCategory === "skills") setSkillGroups([emptySkillGroup()]);
    setEditingMaterialId("");
  };
  const updateStructuredField = (field, value) => {
    setStructuredFields((current) => ({ ...current, [field]: value }));
  };
  const updateSkillGroup = (index, field, value) => {
    setSkillGroups((current) => current.map((group, groupIndex) => groupIndex === index ? { ...group, [field]: value } : group));
  };
  const addSkillGroup = () => setSkillGroups((current) => [...current, emptySkillGroup()]);
  const removeSkillGroup = (index) => setSkillGroups((current) => current.length <= 1 ? [emptySkillGroup()] : current.filter((_, groupIndex) => groupIndex !== index));
  const saveStructured = () => {
    const cleaned = Object.fromEntries(activeStructured.fields.map(([field, , type]) => [field, type === "textarea" ? (structuredFields[field] || "") : String(structuredFields[field] || "").trim()]));
    if (structuredCategory === "skills") {
      const normalizedGroups = skillGroups.map((group) => ({
        title: String(group.title || "").trim(),
        content: String(group.content || "").trim(),
      })).filter((group) => group.title || group.content);
      if (!normalizedGroups.length || normalizedGroups.some((group) => !group.title || !group.content)) {
        notify("请完整填写每组的标题和个人技能/其他内容");
        return Promise.resolve();
      }
      cleaned.content = serializeSkillGroups(normalizedGroups);
      cleaned.groups = normalizedGroups;
    }
    const missing = activeStructured.fields.filter(([field, label, , required]) => required && !String(cleaned[field] || "").trim()).map(([, label]) => label);
    if (missing.length) {
      notify("请填写：" + missing.join("、"));
      return Promise.resolve();
    }
    return runTask(editingMaterialId ? "正在更新结构化经历..." : "正在保存结构化经历...", () => api("/api/career/materials/structured", { method: "POST", body: jsonBody({ category: structuredCategory, fields: cleaned, groups: structuredCategory === "skills" ? cleaned.groups : undefined, replace_material_id: editingMaterialId }) }), [queryKeys.library, queryKeys.facts, queryKeys.workspace, queryKeys.artifacts]).then((data) => {
      resetStructuredForm();
      notify("经历已保存，生成 " + ((data.candidate_facts || []).length) + " 条候选事实");
    }).catch(() => {});
  };
  const startEditMaterial = (material) => runTask("正在读取结构化经历...", () => api("/api/career/materials/" + encodeURIComponent(material.material_id)), []).then((data) => {
    const materialCategory = data.material?.category || material.category;
    if (!STRUCTURED_MATERIAL_CONFIG[materialCategory]) return notify("该材料不支持结构化修改");
    setStructuredCategory(materialCategory);
    const materialText = data.text || "";
    setStructuredFields({ ...emptyStructuredFields(materialCategory), ...parseStructuredMaterialText(materialCategory, materialText) });
    setSkillGroups(materialCategory === "skills" ? parseSkillGroups(materialText) : []);
    setEditingMaterialId(material.material_id);
    notify("已载入，修改后点击保存");
  }).catch(() => notify("材料读取失败"));
  const upload = async () => {
    if (!files.length) return notify("请选择材料文件");
    if (category === "project") {
      const unsupported = files.filter((file) => !PROJECT_ARCHIVE_EXTENSIONS.has(fileExtension(file.name)));
      if (unsupported.length) {
        const knowledgeLike = unsupported.some((file) => INTERVIEW_KNOWLEDGE_EXTENSIONS.has(fileExtension(file.name)));
        notify(knowledgeLike ? "当前选择的是项目经历入口；八股 PDF/Markdown/DOCX/TXT 请切换到“八股资料”后导入。" : "项目经历入口只接受项目 ZIP 包。");
        return;
      }
    }
    await runTask("正在导入 " + files.length + " 份材料...", async () => {
      if (category === "interview_knowledge") {
        const selectedPdfModes = Object.fromEntries(files
          .filter((file) => fileExtension(file.name) === ".pdf")
          .map((file) => [file.webkitRelativePath || file.name, pdfModes[file.webkitRelativePath || file.name] || "normal"]));
        const body = uploadBody({
          files,
          name: files.length === 1 ? files[0].name.replace(/\.[^.]+$/, "") : "",
          domains: "general",
          pdf_modes: JSON.stringify(selectedPdfModes),
        });
        const data = await api("/api/career/interview-knowledge/batch", { method: "POST", body });
        return { category, data };
      }
      const isProject = category === "project";
      const results = await Promise.allSettled(files.map(async (file) => {
        const label = file.name.replace(/\.[^.]+$/, "");
        const body = uploadBody(isProject ? { file, project_id: label, label } : { file, category, label });
        const imported = await api(isProject ? "/api/career/projects/import" : "/api/career/materials", { method: "POST", body });
        if (isProject) {
          const projectId = imported?.project_source?.project_id;
          if (!projectId) throw new Error("项目 " + label + " 导入成功但未返回 project_id，无法自动分析");
          if (imported?.analysis_status !== "running") void requestProjectAnalysis(projectId, true).catch(() => undefined);
        }
        return imported;
      }));
      const failed = results.filter((item) => item.status === "rejected");
      if (failed.length) throw new Error(`有 ${failed.length} 个文件导入失败`);
      return { category, queuedProjectCount: isProject ? files.length : 0 };
    }, [queryKeys.library, queryKeys.projects, queryKeys.facts, queryKeys.workspace, queryKeys.artifacts]).then((result) => {
      setFiles([]);
      setPdfModes({});
      notify(result?.category === "project" ? "项目已导入，后台正在分析 " + (result.queuedProjectCount || 0) + " 个项目；完成后请确认候选事实" : result?.category === "interview_knowledge" ? interviewKnowledgeImportSummary(result.data) : "原始简历已导入候选事实区");
    }).catch((error) => {
      notify(category === "interview_knowledge" ? interviewKnowledgeImportError(error) : (error.message || "导入失败"));
    });
  };
  const skillEditor = structuredCategory === "skills" ? <div className="sa-skill-groups">{skillGroups.map((group, index) => <div className="sa-skill-group" key={index}><div className="sa-skill-group-head"><strong>分组 {index + 1}</strong>{skillGroups.length > 1 && <button type="button" className="sa-icon-button" title="删除分组" aria-label="删除分组" onClick={() => removeSkillGroup(index)}><Trash2 size={15} /></button>}</div><label>标题<input value={group.title} onChange={(event) => updateSkillGroup(index, "title", event.target.value)} placeholder="例如：技术能力" /></label><label className="sa-field-wide">个人技能/其他<textarea rows={5} value={group.content} onChange={(event) => updateSkillGroup(index, "content", event.target.value)} placeholder="填写该分组的技能、证书或其他内容" /></label></div>)}<button type="button" className="sa-button ghost" onClick={addSkillGroup}><Plus size={15} />新增分组</button></div> : null;
  return <section className="sa-panel-grid">
    <div className="sa-card sa-library-summary"><div className="sa-card-head"><div><h2>个人材料库</h2><p>长期保存，与单次 JD 和简历会话分离；八股资料会进入面试知识库。</p></div><span className="sa-badge">版本 {shortId(library?.material_catalog_version)}</span></div><div className="sa-stat-grid">{[["材料", library?.summary?.material_count], ["项目", library?.summary?.project_count], ["知识块", library?.summary?.interview_knowledge_chunk_count], ["待确认", library?.summary?.pending_fact_count]].map(([label, value]) => <div key={label}><strong>{value || 0}</strong><span>{label}</span></div>)}</div><div className="sa-upload-row"><select value={category} onChange={(event) => { setCategory(event.target.value); setFiles([]); setPdfModes({}); }}>{UPLOAD_CATEGORY_OPTIONS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select><label className="sa-upload"><Upload size={15} />选择文件<input type="file" multiple accept={UPLOAD_ACCEPTS[category] || UPLOAD_ACCEPTS.resume} onChange={(event) => { const nextFiles = [...event.target.files]; setFiles(nextFiles); setPdfModes(Object.fromEntries(nextFiles.filter((file) => fileExtension(file.name) === ".pdf").map((file) => [file.webkitRelativePath || file.name, "normal"]))); }} /></label><button className="sa-button primary" onClick={upload}>导入文件</button>{files.length > 0 && <span className="sa-inline-note">已选 {files.length} 个文件</span>}</div>{category === "interview_knowledge" && files.some((file) => fileExtension(file.name) === ".pdf") && <div className="sa-pdf-mode-list">{files.filter((file) => fileExtension(file.name) === ".pdf").map((file) => { const key = file.webkitRelativePath || file.name; return <div className="sa-pdf-mode-row" key={key}><span title={key}>{key}</span><select value={pdfModes[key] || "normal"} onChange={(event) => setPdfModes((current) => ({ ...current, [key]: event.target.value }))} aria-label={`${key} 的 PDF 类型`}><option value="normal">普通 PDF</option><option value="qa">问答 PDF</option></select></div>; })}</div>}</div>
    <div className="sa-card"><div className="sa-card-head"><div><h2>待确认个人事实</h2><p>只有确认后的事实才会进入简历证据。</p></div><span className="sa-badge orange">{facts.length} 条</span></div>{facts.length ? <><div className="sa-fact-list">{facts.slice(0, 80).map((fact) => <div className="sa-fact" key={fact.candidate_fact_id}><div><strong>{fact.label}</strong><p>{fact.value}</p><small>{(fact.source_refs || []).map((source) => source.excerpt || source.source_id).join("；")}</small></div><span className="sa-badge orange">候选</span></div>)}</div><div className="sa-actions"><button className="sa-button primary" onClick={() => confirmFacts("confirm")}><Check size={15} />全部确认</button><button className="sa-button ghost" onClick={() => confirmFacts("reject")}><X size={15} />全部拒绝</button></div></> : <EmptyState text="暂无待确认事实，填写经历或导入原始简历后会在这里出现。" />}</div>
    <div className="sa-card sa-wide sa-manual-card"><div className="sa-card-head"><div><h2>手动填写简历信息</h2><p>教育、实习、技能、科研和获奖经历在前端直接填写；保存后进入待确认事实区。</p></div><span className="sa-badge green">{MATERIAL_CATEGORY_LABELS[structuredCategory]}</span></div><div className="sa-material-tabs">{STRUCTURED_CATEGORY_OPTIONS.map((id) => <button key={id} type="button" className={"sa-tab-button " + (structuredCategory === id ? "active" : "")} onClick={() => chooseStructuredCategory(id)}><span>{MATERIAL_CATEGORY_LABELS[id]}</span><b>{categoryCounts[id] || 0}</b></button>)}</div><div className="sa-structured-layout"><div className="sa-structured-form"><h3>{activeStructured.title}</h3><p>{activeStructured.note}</p>{skillEditor || <div className="sa-form-grid">{activeStructured.fields.map(([field, label, fieldType, required, options]) => <label key={field} className={fieldType === "textarea" ? "sa-field-wide" : ""}>{label}{required ? " *" : ""}{fieldType === "textarea" ? <textarea rows={4} value={structuredFields[field] || ""} onChange={(event) => updateStructuredField(field, event.target.value)} placeholder={label} /> : fieldType === "select" ? <select value={structuredFields[field] || ""} onChange={(event) => updateStructuredField(field, event.target.value)}><option value="">请选择{label}</option>{(options || []).map((option) => <option value={option} key={option}>{option}</option>)}</select> : <input type={fieldType === "date" ? "date" : "text"} value={structuredFields[field] || ""} onChange={(event) => updateStructuredField(field, event.target.value)} placeholder={label} />}</label>)}</div>}<div className="sa-actions"><button className="sa-button primary" onClick={saveStructured}>{editingMaterialId ? "保存修改" : "保存经历"}</button><button className="sa-button ghost" onClick={resetStructuredForm}>清空</button>{editingMaterialId && <span className="sa-inline-note">正在修改 {shortId(editingMaterialId)}</span>}</div></div><div className="sa-structured-existing"><h3>已填写的{MATERIAL_CATEGORY_LABELS[structuredCategory]}</h3>{activeStructuredMaterials.length ? activeStructuredMaterials.slice(0, 12).map((material) => <div className="sa-row" key={material.material_id}><div><strong>{material.label}</strong><small>修订 {material.revision || 1} · {shortId(material.material_id)}</small></div><button className="sa-button ghost" onClick={() => startEditMaterial(material)}>修改</button></div>) : <EmptyState text={"暂无" + MATERIAL_CATEGORY_LABELS[structuredCategory] + "，可以直接在左侧填写。"} />}</div></div></div>
    <div className="sa-card sa-wide">
      <div className="sa-card-head">
        <div><h2>材料与项目目录</h2><p>显示已保存材料和项目；结构化经历可在上方对应模块修改。</p></div>
        <div className="sa-card-head-actions">
          <label className="sa-select-all"><input ref={selectAllMaterialsRef} className="sa-checkbox" type="checkbox" checked={allMaterialsSelected} onChange={toggleAllMaterials} disabled={!materials.length} aria-label="全选材料" />全选材料</label>
          <label className="sa-select-all"><input ref={selectAllProjectsRef} className="sa-checkbox" type="checkbox" checked={allProjectsSelected} onChange={toggleAllProjects} disabled={!projectCatalogEntries.length} aria-label="全选项目" />全选项目</label>
          <span className="sa-badge">{materials.length} 份材料</span>
          <span className="sa-badge">{projectCatalogEntries.length} 个项目</span>
          <button type="button" className="sa-button ghost" disabled={!apiCompatible || !selectedMaterialIds.size} title={apiCompatible ? "删除已勾选的材料" : "后端版本不兼容，请先重启 Smart Agent"} onClick={deleteSelectedMaterials}><Trash2 size={15} />删除材料（{selectedMaterialIds.size}）</button>
          <button type="button" className="sa-button ghost" disabled={!apiCompatible || !selectedProjectIds.size} title={apiCompatible ? "删除已勾选的项目" : "后端版本不兼容，请先重启 Smart Agent"} onClick={deleteSelectedProjects}><Trash2 size={15} />删除项目（{selectedProjectIds.size}）</button>
        </div>
      </div>
      <div className="sa-material-list">
        {materials.slice(0, 80).map((material) => <div className="sa-row" key={material.material_id}><input className="sa-checkbox" type="checkbox" checked={selectedMaterialIds.has(material.material_id)} onChange={() => toggleMaterial(material.material_id)} aria-label={`选择材料 ${material.label || material.material_id}`} /><div><strong>{material.label}</strong><small>{MATERIAL_CATEGORY_LABELS[material.category] || material.category} · 修订 {material.revision || 1}</small></div><span>{shortId(material.material_id)}</span><button type="button" className="sa-icon-button" title="删除材料" aria-label={`删除材料 ${material.label || material.material_id}`} onClick={() => deleteMaterial(material)}><Trash2 size={15} /></button></div>)}
        {projectCatalogEntries.slice(0, 80).map(({ project, analysis, analysisStatus, latestAnalysisRun }) => {
          const statusLabels = { running: "分析中", pending: "待分析", failed: "分析失败", incomplete: "分析不完整", completed: "已完成", cached: "已完成" };
          const statusLabel = statusLabels[analysisStatus] || analysisStatus || "待分析";
          return <div className="sa-row" key={project.project_id}>
            <input className="sa-checkbox" type="checkbox" checked={selectedProjectIds.has(project.project_id)} onChange={() => toggleProject(project.project_id)} aria-label={`选择项目 ${project.display_name || project.project_id}`} />
            <div>
              <strong>{project.display_name || project.project_id}</strong>
              <small>project · {project.status || "registered"} · {project.artifact_count || 0} 个可读文件 · {statusLabel}</small>
              {analysisStatus === "failed" && latestAnalysisRun?.error && <small className="sa-project-analysis-error">{latestAnalysisRun.error}</small>}
            </div>
            <button className="sa-button" disabled={analysisStatus === "running"} onClick={() => api("/api/career/project-library/" + encodeURIComponent(project.project_id)).then((data) => notify(data.latest_analysis ? "项目分析已读取" : (data.analysis_status === "running" ? "项目正在分析" : "项目尚未分析"))).catch(() => notify("项目读取失败"))}>{analysis ? "已分析" : analysisStatus === "running" ? "分析中" : "查看"}</button>
            <button type="button" className="sa-icon-button" title="删除项目" aria-label={`删除项目 ${project.display_name || project.project_id}`} onClick={() => deleteProject(project)}><Trash2 size={15} /></button>
          </div>;
        })}
        {projectCatalogEntries.some(({ analysis, analysisStatus }) => !analysis && analysisStatus !== "running") && <div className="sa-project-pending-actions"><span>以下项目尚未生成分析结果：</span>{projectCatalogEntries.filter(({ analysis, analysisStatus }) => !analysis && analysisStatus !== "running").map(({ project, analysisStatus }) => <button type="button" className="sa-button ghost" key={project.project_id + "-analyze"} onClick={() => analyzeProject(project)}>{analysisStatus === "failed" ? "重试" : "分析"} {project.display_name || project.project_id}</button>)}</div>}
        {projectCatalogEntries
          .filter(({ analysis, description }) => description?.status === "needs_user_confirmation" || (analysis && !description))
          .map(({ project, analysis, description }) => <ProjectAnalysisReview key={project.project_id + "-review"} project={project} analysis={analysis} description={description} onConfirm={confirmProjectDescription} />)}
        {!materials.length && !projectCatalogEntries.length && <EmptyState text="尚未导入材料或登记项目。" />}
      </div>
    </div>
    <div className="sa-card sa-wide"><div className="sa-card-head"><div><h2>八股资料目录</h2><p>删除资料会同步清理知识块、向量索引和图索引；不会删除电脑上的原始文件。</p></div><div className="sa-card-head-actions"><label className="sa-select-all"><input ref={selectAllKnowledgeRef} className="sa-checkbox" type="checkbox" checked={allKnowledgeSelected} onChange={toggleAllKnowledge} disabled={!knowledgeSources.length} aria-label="全选八股资料" />全选</label><span className="sa-badge">{knowledgeSources.length} 份</span><button type="button" className="sa-button ghost" disabled={!apiCompatible || !selectedKnowledgeIds.size} title={apiCompatible ? "删除已勾选的八股资料" : "后端版本不兼容，请先重启 Smart Agent"} onClick={deleteSelectedKnowledge}><Trash2 size={15} />删除已勾选（{selectedKnowledgeIds.size}）</button></div></div><div className="sa-material-list">{knowledgeSources.slice(0, 80).map((source) => <div className="sa-row" key={source.source_id}><input className="sa-checkbox" type="checkbox" checked={selectedKnowledgeIds.has(source.source_id)} onChange={() => toggleKnowledge(source.source_id)} aria-label={`选择八股资料 ${source.name || source.source_id}`} /><div><strong>{source.name || source.source_id}</strong><small>{source.source_type || "interview knowledge"} · {(source.corpus || {}).chunk_count || source.chunk_count || 0} 个知识块</small></div><span>{shortId(source.source_id)}</span><button type="button" className="sa-icon-button" disabled={!apiCompatible} title={apiCompatible ? "删除八股资料" : "后端版本不兼容，请先重启 Smart Agent"} aria-label={`删除八股资料 ${source.name || source.source_id}`} onClick={() => deleteKnowledge(source)}><Trash2 size={15} /></button></div>)}{!knowledgeSources.length && <EmptyState text="尚未导入八股资料。" />}</div></div>
  </section>;
}

function EvidencePanel({ context, runTask, onSuccess, notify }) {
  const [text, setText] = useState(context.jd?.raw_text || "");
  const [role, setRole] = useState(context.jd?.role_title || "");
  const [company, setCompany] = useState(context.jd?.company || "");
  const [ocrMessage, setOcrMessage] = useState("");
  useEffect(() => { setText(context.jd?.raw_text || ""); setRole(context.jd?.role_title || ""); setCompany(context.jd?.company || ""); }, [context.jd?.jd_id]);
  const analyze = () => runTask("正在保存并分析岗位 JD...", () => api("/api/career/jd/analyze", { method: "POST", body: jsonBody({ text, role_title: role, company }) }), [queryKeys.workspace, queryKeys.artifacts]).then((data) => { onSuccess(data); notify("岗位 JD 已保存"); }).catch(() => {});
  const ocr = async (file) => { if (!file) return; setOcrMessage("正在识别图片..."); try { const data = await api("/api/career/ocr", { method: "POST", body: uploadBody({ file }) }); setText(data.candidate?.text || ""); setOcrMessage("OCR 已生成候选文本，请校对后提交。"); } catch (error) { setOcrMessage(error.message || "OCR 失败"); } };
  return <section className="sa-panel-grid"><div className="sa-card sa-wide"><div className="sa-card-head"><div><h2>岗位 JD</h2><p>JD 与当前简历会话绑定；历史岗位不会进入新会话。</p></div><span className={`sa-badge ${context.jd ? "green" : "orange"}`}>{context.jd ? `${context.requirements.length} 项要求` : "待提交"}</span></div><div className="sa-form-grid"><label>岗位名称<input value={role} onChange={(event) => setRole(event.target.value)} placeholder="例如：Agent 开发工程师" /></label><label>公司<input value={company} onChange={(event) => setCompany(event.target.value)} placeholder="目标公司" /></label></div><label className="sa-field">岗位描述<textarea value={text} onChange={(event) => setText(event.target.value)} rows={16} placeholder="粘贴岗位 JD，或使用左侧材料导入。" /></label><div className="sa-actions"><button className="sa-button primary" onClick={analyze} disabled={!text.trim()}><ShieldCheck size={15} />保存并分析</button><label className="sa-upload"><Upload size={15} />图片 OCR<input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => ocr(event.target.files?.[0])} /></label>{ocrMessage && <span className="sa-inline-note">{ocrMessage}</span>}</div></div><div className="sa-card"><div className="sa-card-head"><div><h2>结构化岗位要求</h2><p>只展示当前会话的要求。</p></div></div>{context.requirements.length ? <div className="sa-requirements">{context.requirements.map((item, index) => <div className="sa-row" key={item.requirement_id}><b>{String(index + 1).padStart(2, "0")}</b><span>{item.text || item.original_text}</span><em>{item.category || "requirement"}</em></div>)}</div> : <EmptyState text="提交 JD 后显示结构化要求。" />}</div></section>;
}

function ResumePanel({ context, templates, records, runTask, notify, onNewResume }) {
  const queryClient = useQueryClient();
  const [templateId, setTemplateId] = useState("");
  const [fullName, setFullName] = useState(context.document?.header?.full_name || "");
  const [headline, setHeadline] = useState(context.document?.header?.headline || "");
  const [email, setEmail] = useState(context.document?.header?.email || "");
  const [phone, setPhone] = useState(context.document?.header?.phone || "");
  const [portfolioLink, setPortfolioLink] = useState(context.document?.header?.links?.[0] || "");
  const [selectedVersionId, setSelectedVersionId] = useState("");
  const [selectingCurrentId, setSelectingCurrentId] = useState("");
  const [deletingVersionId, setDeletingVersionId] = useState("");
  const [customizationInput, setCustomizationInput] = useState("");
  const runId = context.run?.resume_run_id;
  const versions = resumeVersions(records, context.run);
  const currentDocumentId = context.document?.resume_document_id || "";
  const latestVersionId = versions[0]?.resume_document_id || currentDocumentId;
  useEffect(() => {
    setFullName(context.document?.header?.full_name || "");
    setHeadline(context.document?.header?.headline || "");
    setEmail(context.document?.header?.email || "");
    setPhone(context.document?.header?.phone || "");
    setPortfolioLink(context.document?.header?.links?.[0] || "");
  }, [context.document?.resume_document_id]);
  useEffect(() => {
    setSelectedVersionId(currentDocumentId || latestVersionId || "");
  }, [runId, currentDocumentId, latestVersionId]);
  if (!runId) return <EmptyPanel icon={<FileText size={30} />} title="还没有岗位会话" text="先在“岗位 JD”中提交目标岗位，再生成专属简历。" />;
  const deliver = () => runTask(
    context.document ? "正在按定制要求更新、核验并生成 PDF..." : "正在定制、核验并生成 PDF...",
    () => api("/api/career/resume/deliver", {
      method: "POST",
      body: jsonBody({
        resume_run_id: runId,
        template_id: templateId,
        customization_request: customizationInput.trim(),
        header: { full_name: fullName, headline, email, phone, links: portfolioLink.trim() ? [portfolioLink.trim()] : [] },
      }),
    }),
    [queryKeys.artifacts, queryKeys.workspace],
  ).then((data) => {
    setCustomizationInput("");
    setSelectedVersionId(data.generation?.document?.resume_document_id || "");
    notify(data.status === "delivered" ? "简历已生成、核验并导出 PDF" : "简历已生成，核验结果需要处理");
  }).catch(() => {});
  const selectedVersion = versions.find((version) => version.resume_document_id === selectedVersionId) || context.document || versions[0];
  const selectedVersionPdfUrl = selectedVersion?.resume_document_id
    ? `/api/career/resume/documents/${encodeURIComponent(selectedVersion.resume_document_id)}/pdf?v=${encodeURIComponent(selectedVersion.updated_at || selectedVersion.revision || selectedVersion.resume_document_id)}`
    : "";
  const runPdfUrl = context.run?.rendered_artifacts?.length ? `/api/career/resume/runs/${encodeURIComponent(runId)}/pdf?v=${encodeURIComponent(context.run.rendered_artifacts.at(-1))}` : "";
  const pdfUrl = selectedVersionPdfUrl || runPdfUrl;
  const selectedTemplate = templates.find((template) => template.template_id === templateId) || templates[0] || null;
  const customizationMessages = context.run?.customization_messages || [];
  const selectVersion = (version, index) => {
    const documentId = version.resume_document_id;
    if (!documentId) return;
    setSelectedVersionId(documentId);
    notify(`已切换到版本 ${versions.length - index}`);
  };
  const makeCurrentVersion = () => {
    const documentId = selectedVersion?.resume_document_id;
    if (!documentId || documentId === currentDocumentId || selectingCurrentId) return;
    setSelectingCurrentId(documentId);
    runTask(
      "正在设置当前简历版本...",
      () => api("/api/career/resume/select", { method: "POST", body: jsonBody({ resume_run_id: runId, resume_document_id: documentId }) }),
    ).then((data) => {
      queryClient.setQueryData(queryKeys.artifacts, (current) => ({
        ...(current || {}),
        records: mergeCareerRecords(current?.records || [], [data.resume_run, data.resume_document]),
      }));
      notify("已设为当前简历版本");
    }).catch(() => {}).finally(() => setSelectingCurrentId(""));
  };
  const deleteVersion = (version) => {
    const documentId = version?.resume_document_id;
    if (!documentId || deletingVersionId) return;
    const isCurrent = documentId === currentDocumentId;
    const versionLabel = versions.length - versions.findIndex((item) => item.resume_document_id === documentId);
    if (!window.confirm(`确定删除${isCurrent ? "当前" : ""}简历版本 ${versionLabel} 吗？${isCurrent ? "删除后将自动切换到其他版本；若无其他版本则清空当前简历。" : ""}`)) return;
    setDeletingVersionId(documentId);
    runTask(
      "正在删除简历版本...",
      () => api(`/api/career/resume/versions/${encodeURIComponent(documentId)}`, { method: "DELETE", body: jsonBody({ resume_run_id: runId, resume_document_id: documentId }) }),
      [queryKeys.artifacts, queryKeys.workspace],
    ).then((data) => {
      const replacementId = data?.replacement_document?.resume_document_id || data?.resume_run?.resume_document_ref?.record_id || "";
      if (selectedVersionId === documentId) setSelectedVersionId(replacementId || versions.find((item) => item.resume_document_id !== documentId)?.resume_document_id || "");
      notify("简历版本已删除");
    }).catch(() => {}).finally(() => setDeletingVersionId(""));
  };
  return (
    <section className="sa-resume-layout">
      <div className="sa-card sa-resume-form">
        <div className="sa-card-head">
          <div>
            <h2>简历生成</h2>
            <p>{context.jd?.company || "目标公司"} · {context.jd?.role_title || "目标岗位"}</p>
          </div>
          <span className={`sa-badge ${context.run?.status === "rendered" ? "green" : "orange"}`}>
            {context.run?.status === "rendered" ? "已交付" : context.document ? "待处理" : "待生成"}
          </span>
        </div>
        <label>姓名<input value={fullName} onChange={(event) => setFullName(event.target.value)} placeholder="简历姓名" /></label>
        <label>标题<input value={headline} onChange={(event) => setHeadline(event.target.value)} placeholder="例如：Agent 开发工程师" /></label>
        <label>邮箱<input value={email} onChange={(event) => setEmail(event.target.value)} placeholder="name@example.com" /></label>
        <label>电话<input value={phone} onChange={(event) => setPhone(event.target.value)} placeholder="联系电话" /></label>
        <label>作品链接<input value={portfolioLink} onChange={(event) => setPortfolioLink(event.target.value)} placeholder="https://..." /></label>
        <label>模板<select value={templateId} onChange={(event) => setTemplateId(event.target.value)}><option value="">默认模板（自动）</option>{templates.map((template) => <option key={template.template_id} value={template.template_id}>{template.name || template.template_id}</option>)}</select></label>
        <TemplatePreview template={selectedTemplate} isAuto={!templateId} />
        {context.document && <div className="sa-actions column sa-resume-primary-actions"><button type="button" className="sa-button ghost sa-resume-deliver" onClick={onNewResume}><Plus size={15} />创建新简历</button></div>}
        <div className="sa-version-list">
          <h3>历史版本</h3>
          {versions.slice(0, 20).map((version, index) => {
            const isCurrent = version.resume_document_id === currentDocumentId;
            return <div className={`sa-row sa-version-row ${version.resume_document_id === selectedVersion?.resume_document_id ? "active" : ""}`} key={version.resume_document_id}><button type="button" className="sa-version-button" onClick={() => selectVersion(version, index)}><span>版本 {versions.length - index}</span><small>{formatDate(version.updated_at)}</small></button>{isCurrent && <em>当前</em>}<button type="button" className="sa-icon-button sa-version-delete" title={isCurrent ? "删除当前简历" : "删除简历版本"} aria-label={`${isCurrent ? "删除当前简历" : "删除简历版本"} ${versions.length - index}`} onClick={() => deleteVersion(version)} disabled={Boolean(deletingVersionId)}><Trash2 size={14} /></button></div>;
          })}
          {!versions.length && <small>暂无版本</small>}
        </div>
      </div>
      <div className="sa-card sa-preview">
        <div className="sa-card-head">
          <div>
            <h2>预览</h2>
            <p>{selectedVersion ? `正在预览版本 ${versions.length - versions.findIndex((version) => version.resume_document_id === selectedVersion.resume_document_id)}。` : "Agent 生成、核验并导出后在此显示。"}</p>
          </div>
          <div className="sa-actions">
            {selectedVersion?.resume_document_id && selectedVersion.resume_document_id !== currentDocumentId && <button type="button" className="sa-button" onClick={makeCurrentVersion} disabled={Boolean(selectingCurrentId)}><Check size={15} />设为当前</button>}
            <button className="sa-icon-button" onClick={() => window.open(pdfUrl || "about:blank", "_blank")} disabled={!pdfUrl} title="新窗口打开"><ChevronRight size={17} /></button>
          </div>
        </div>
        {pdfUrl ? <iframe title="简历 PDF 预览" src={pdfUrl} /> : <EmptyState text="生成后会自动完成核验和 PDF 导出。" />}
        <div className="sa-resume-agent">
          <div className="sa-resume-agent-head"><MessageCircle size={16} /><div><h3>简历定制 Agent</h3><p>{context.document ? "提出修改要求，Agent 会生成新的可切换版本。" : "告诉 Agent 你想重点呈现的项目或经历。"}</p></div></div>
          <div className="sa-resume-agent-thread" aria-live="polite">
            {customizationMessages.map((message, index) => <div key={`${message.created_at || index}-${index}`} className={`sa-resume-agent-message ${message.role === "user" ? "user" : "assistant"}`}><span>{message.role === "user" ? "你" : "Agent"}</span><p>{message.content}</p></div>)}
            {!customizationMessages.length && <div className="sa-resume-agent-empty">{context.document ? "本版本已由 Agent 完成生成、核验和 PDF 导出；可以继续提出修改要求生成新版本。" : "例如：优先突出 Smart Agent 项目，简化不相关经历。"}</div>}
          </div>
          <textarea rows={3} value={customizationInput} onChange={(event) => setCustomizationInput(event.target.value)} maxLength={1200} placeholder={context.document ? "例如：突出项目中的工程化设计，内容更简洁" : "例如：优先突出最相关的项目和个人贡献"} />
          <button className="sa-button primary sa-resume-deliver" onClick={deliver} disabled={!fullName}><Send size={15} />{context.document ? "按此修改并更新 PDF" : "生成并导出 PDF"}</button>
        </div>
      </div>
    </section>
  );
}

function TemplatePreview({ template, isAuto }) {
  if (!template) {
    return <div className="sa-template-preview empty">暂无可预览模板</div>;
  }
  const previewImage = template.preview_image_url;
  const previewPdf = template.preview_url;
  const name = template.name || template.original_name || template.template_id;
  return (
    <div className="sa-template-preview">
      <div className="sa-template-preview-head">
        <span>{isAuto ? "自动使用" : "当前模板"}：{name}</span>
        {previewPdf && <button type="button" className="sa-icon-button" onClick={() => window.open(previewPdf, "_blank")} title="打开模板 PDF"><ChevronRight size={15} /></button>}
      </div>
      {previewImage ? (
        <img src={previewImage} alt={`${name} 模板预览`} />
      ) : (
        <div className="sa-template-preview-empty">当前模板没有可预览 PDF。</div>
      )}
    </div>
  );
}

function InterviewCoachPanel({ context, runTask, notify }) {
  const workspace = useSession();
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  useEffect(() => {
    const last = workspace.coachMessages.at(-1);
    setResult(last?.role === "assistant" ? { answer: last.content } : null);
  }, [workspace.workspaceSessionId, workspace.activeRunId, workspace.inputBundleId]);
  const ask = () => {
    const text = question.trim();
    if (!text || workspace.pending || !workspace.workspaceSessionId) return;
    runTask("正在检索面试辅导资料...", () => api("/api/career/interview-coach/ask", {
      method: "POST",
      body: jsonBody({
        question: text,
        session_id: workspace.workspaceSessionId,
        context_revision: workspace.contextRevision,
        resume_run_id: context.run?.resume_run_id || "",
        top_k: 6,
      }),
    })).then(async (data) => {
      await workspace.refresh();
      setResult(data);
      notify("面试辅导答案已生成");
    }).catch(() => {});
  };
  const sources = result?.sources || {};
  const warnings = Array.isArray(result?.warnings) ? result.warnings : [];
  const suggestions = Array.isArray(result?.follow_up_suggestions) ? result.follow_up_suggestions : [];
  return (
    <section className="sa-coach-layout">
      <div className="sa-card sa-coach-main">
        <div className="sa-card-head">
          <div>
            <h2>面试辅导</h2>
            <p>自动路由到八股知识、导入项目库和项目 RAG，适合问概念、项目或混合问题。</p>
          </div>
          <span className="sa-badge">{context.jd?.role_title || "通用辅导"}</span>
        </div>
        <label className="sa-field">学生问题<textarea rows={7} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="例如：结合我的 RAG 项目讲一下向量召回和重排怎么回答？" /></label>
        <div className="sa-actions">
          <button className="sa-button primary" type="button" onClick={ask} disabled={!question.trim()}><Send size={15} />生成辅导答案</button>
        </div>
        {result ? (
          <div className="sa-coach-answer">
            <div className="sa-section-title"><strong>辅导答案</strong><span>{COACH_TYPE_LABELS[result.question_type] || result.question_type}</span></div>
            <p>{result.answer || "未生成答案。"}</p>
            {suggestions.length > 0 && <div className="sa-coach-suggestions"><strong>可继续追问</strong>{suggestions.map((item, index) => <span key={index}>{item}</span>)}</div>}
            {warnings.length > 0 && <div className="sa-coach-warnings">{warnings.map((item, index) => <span key={index}>{item}</span>)}</div>}
          </div>
        ) : <EmptyState text="输入问题后，会先判断类型并检索对应知识来源。" />}
      </div>
      <div className="sa-card">
        <div className="sa-card-head">
          <div>
            <h2>检索来源</h2>
            <p>展示八股知识、项目 RAG、真实项目底层文件和项目登记信息。</p>
          </div>
        </div>
        <InterviewCoachSourceList title="八股知识库" items={sources.interview_knowledge} type="knowledge" />
        <InterviewCoachSourceList title="项目 RAG" items={sources.projects} type="project_hit" />
        <InterviewCoachSourceList title="项目底层文件" items={sources.project_files} type="project_file" />
        <InterviewCoachSourceList title="导入项目库" items={sources.project_sources} type="project_source" />
      </div>
    </section>
  );
}

function InterviewCoachSourceList({ title, items, type }) {
  const list = Array.isArray(items) ? items : [];
  return (
    <div className="sa-coach-source-list">
      <div className="sa-section-title"><strong>{title}</strong><span>{list.length} 条</span></div>
      {list.length ? list.map((item, index) => {
        const key = item.source_id || item.chunk_id || item.project_id || String(index);
        if (type === "project_source") {
          return <article key={key}><strong>{item.display_name || item.project_id || "未命名项目"}</strong><small>{item.status || "registered"} · {item.artifact_count || 0} 个文件 · 修订 {item.revision || 1}</small><p>{item.source_type || "project"}</p></article>;
        }
        if (type === "project_hit") {
          return <article key={key}><strong>{item.project_id || "项目片段"}</strong><small>{shortId(item.chunk_id || item.source_id || "")} · lexical {Number(item.lexical_score || 0).toFixed(2)}</small><p>{item.content || "无片段内容"}</p></article>;
        }
        if (type === "project_file") {
          return <article key={key}><strong>{item.display_name || item.project_id || "项目文件"}</strong><small>{item.path || "未命名文件"} · {item.kind || "file"} · lexical {Number(item.lexical_score || 0).toFixed(2)}</small><p>{item.content || "无片段内容"}</p></article>;
        }
        return <article key={key}><strong>{item.topic || item.subdomain || "八股知识"}</strong><small>{item.corpus_id || ""} · {shortId(item.chunk_id || item.source_id || "")}</small><p>{item.text || "无片段内容"}</p></article>;
      }) : <EmptyState text={"暂无" + title + "命中。"} />}
    </div>
  );
}

function InterviewImportPanel() {
  const session = useSession();
  const [jd, setJd] = useState("");
  const [resume, setResume] = useState("");
  const [role, setRole] = useState("");
  const [bundle, setBundle] = useState(null);
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const [error, setError] = useState("");
  useEffect(() => {
    let cancelled = false;
    if (session.inputBundleId) api("/api/career/contexts/" + session.inputBundleId).then(value => {
      if (cancelled) return;
      setJd(value.jd_text); setResume(value.resume_text); setRole(value.role_title); setBundle(value);
    }).catch(failure => { if (!cancelled) setError(failure.message); });
    return () => { cancelled = true; };
  }, [session.inputBundleId]);
  const perform = async (operation) => {
    if (inFlight.current) return;
    inFlight.current = true; setBusy(true); setError("");
    try { await operation(); } catch (failure) { setError(failure.message); }
    finally { inFlight.current = false; setBusy(false); }
  };
  const extract = (file, setter) => {
    if (!file) return;
    perform(async () => {
      if (file.size > 10 * 1024 * 1024) throw new Error("文件不能超过 10 MB");
      const data = await api("/api/career/contexts/extract", { method: "POST", body: uploadBody({ file }) });
      setter(data.text); setBundle(null);
    });
  };
  const confirm = () => perform(async () => {
    let value = bundle;
    if (!value) {
      value = await api("/api/career/contexts/interview-inputs", { method: "POST", body: jsonBody({ jd_text: jd, resume_text: resume, role_title: role }) });
      setBundle(value);
    }
    const confirmed = await api("/api/career/contexts/" + value.session_id + "/confirm", {
      method: "POST", body: jsonBody({ expected_revision: value.context_revision, request_id: crypto.randomUUID() }),
    });
    await session.selectBundle(confirmed.session_id);
  });
  return <section className="sa-interview-inputs">
    <div className="sa-card-head"><h2>本次面试材料</h2><span className="sa-badge">{session.inputBundleId ? "独立导入 · 用户提供" : "当前岗位会话"}</span></div>
    <label className="sa-field">岗位名称<input value={role} maxLength={120} disabled={busy} onChange={e => { setRole(e.target.value); setBundle(null); }} /></label>
    <div className="sa-form-grid">{[["岗位 JD", jd, setJd], ["简历", resume, setResume]].map(([label, value, setter]) => <div key={label}>
      <label className="sa-field">{label}<textarea aria-label={label} rows={6} maxLength={20000} value={value} disabled={busy} onChange={e => { setter(e.target.value); setBundle(null); }} /></label>
      <label className="sa-upload"><Upload size={15} />{label}文件<input type="file" disabled={busy} accept=".txt,.md,.markdown,.pdf,.docx" onChange={e => extract(e.target.files?.[0], setter)} /></label>
    </div>)}</div>
    <div className="sa-actions"><button className="sa-button primary" disabled={busy || session.pending || !session.workspaceSessionId || !jd.trim() || !resume.trim() || !role.trim()} onClick={confirm}><ShieldCheck size={15} />确认并使用这份 JD 与简历</button></div>
    {error && <p role="alert">{error}</p>}
  </section>;
}

function InterviewModePanel({ context, sessions, activeInterview, runTask, onSelect, notify }) {
  const workspace = useSession();
  const queryClient = useQueryClient();
  const startingRef = useRef(false);
  const [answer, setAnswer] = useState("");
  const [voiceQuestion, setVoiceQuestion] = useState("");
  const [deletingInterviewId, setDeletingInterviewId] = useState("");
  const [nextQuestionError, setNextQuestionError] = useState("");
  const [retryingNextQuestion, setRetryingNextQuestion] = useState(false);
  const submitInFlightRef = useRef(false);
  const [mode, setMode] = useState(() => sessionStorage.getItem("smart-agent-interview-mode") || "realistic_simulation");
  const currentInterview = activeInterview && interviewMode(activeInterview) === mode ? activeInterview : null;
  const currentDisplayStatus = currentInterview ? interviewDisplayStatus(currentInterview, sessions) : "待开始";
  const pendingQuestion = currentInterview?.pending_question || null;
  const currentIsLive = currentDisplayStatus === "进行中" && Boolean(pendingQuestion?.question_id);
  const waitingForQuestion = currentInterview && currentDisplayStatus === "进行中" && !currentIsLive;
  const modeMeta = INTERVIEW_MODES.find((item) => item.value === mode) || INTERVIEW_MODES[0];
  useEffect(() => {
    if (!activeInterview?.interview_id) return;
    const nextMode = interviewMode(activeInterview);
    setMode(nextMode);
    sessionStorage.setItem("smart-agent-interview-mode", nextMode);
  }, [activeInterview?.interview_id]);
  const selectMode = (nextMode) => {
    setMode(nextMode);
    sessionStorage.setItem("smart-agent-interview-mode", nextMode);
    setAnswer("");
    setVoiceQuestion("");
    setNextQuestionError("");
  };
  const start = () => {
    if (startingRef.current || workspace.pending || !workspace.workspaceSessionId) return;
    startingRef.current = true;
    return runTask(`正在创建${modeMeta.label}面试...`, () => api("/api/career/interviews/start", {
    method: "POST",
      body: jsonBody({
        mode,
        workspace_session_id: workspace.workspaceSessionId,
        context_revision: workspace.contextRevision,
        resume_run_id: context.run?.resume_run_id || "",
        jd_id: context.jd?.jd_id || "",
        resume_document_id: context.run?.resume_document_ref?.record_id || context.document?.resume_document_id || "",
        resume_document_revision: context.run?.resume_document_ref?.revision || context.document?.revision || null,
      }),
  }), [queryKeys.interviews]).then(async (data) => {
    await workspace.refresh();
    onSelect({ ...data.session, pending_question: data.question });
    setNextQuestionError("");
    notify(`${modeMeta.label}面试已开始`);
  }).catch(async (error) => {
    setNextQuestionError(error.message);
    await workspace.refresh();
    await queryClient.invalidateQueries({ queryKey: queryKeys.interviews });
  }).finally(() => { startingRef.current = false; });
  };
  const requestNextQuestion = async (interviewId, candidates) => {
    const path = candidates?.path;
    const refs = (candidates?.selected_blocks || []).map((item) => item.chunk_id).filter(Boolean);
    const request = () => {
      if (path === "depth") {
        return api(`/api/career/interviews/${interviewId}/question/depth`, {
          method: "POST",
          body: jsonBody({ candidate_knowledge_refs: refs }),
        });
      }
      if (path === "breadth") {
        return api(`/api/career/interviews/${interviewId}/question/breadth`, {
          method: "POST",
          body: jsonBody({ candidate_knowledge_refs: refs }),
        });
      }
      return api(`/api/career/interviews/${interviewId}/question`, { method: "POST" });
    };
    try {
      return await request();
    } catch (error) {
      // The specialized lane can reject stale candidate references while the
      // durable Runtime state is already ready for the next legal question.
      if (!path) throw error;
      try {
        return await api(`/api/career/interviews/${interviewId}/question`, { method: "POST" });
      } catch (fallbackError) {
        throw new Error(fallbackError.message || error.message || "下一题生成失败");
      }
    }
  };
  const submit = () => {
    if (submitInFlightRef.current || !answer.trim() || !currentInterview?.interview_id || !pendingQuestion?.question_id) return;
    submitInFlightRef.current = true;
    setNextQuestionError("");
    runTask("正在处理回答并生成下一题...", async () => {
      const interviewId = encodeURIComponent(currentInterview.interview_id);
      const processed = await api(`/api/career/interviews/${interviewId}/answer/process`, { method: "POST", body: jsonBody({ answer, question_id: pendingQuestion.question_id }) });
      const candidates = processed.candidates || {};
      if (processed.status === "completed" || processed.decision?.action === "end") return processed;
      return requestNextQuestion(interviewId, candidates);
    }, [queryKeys.interviews]).then(() => {
      setAnswer("");
      setNextQuestionError("");
      notify("下一题已生成");
    }).catch((error) => {
      const message = `下一题生成失败：${error.message || "未知错误"}`;
      setNextQuestionError(message);
      notify(message);
    }).finally(() => {
      submitInFlightRef.current = false;
    });
  };
  const retryNextQuestion = () => {
    if (retryingNextQuestion || !currentInterview?.interview_id) return;
    const interviewId = encodeURIComponent(currentInterview.interview_id);
    setRetryingNextQuestion(true);
    setNextQuestionError("");
    runTask("正在重试生成下一题...", () => api(`/api/career/interviews/${interviewId}/question`, { method: "POST" }), [queryKeys.interviews])
      .then(() => {
        setNextQuestionError("");
        notify("下一题已生成");
      })
      .catch((error) => {
        const message = `下一题生成失败：${error.message || "未知错误"}`;
        setNextQuestionError(message);
        notify(message);
      })
      .finally(() => setRetryingNextQuestion(false));
  };
  const deleteInterview = (item) => {
    const interviewId = String(item?.interview_id || "").trim();
    if (!interviewId || deletingInterviewId) return;
    const title = interviewMode(item) === "gap_analysis" ? "查漏补缺" : "真实模拟";
    if (!window.confirm(`确定删除这条${title}面试记录吗？删除后无法恢复。`)) return;
    setDeletingInterviewId(interviewId);
    runTask("正在删除面试记录...", () => api(`/api/career/sessions/interview/${encodeURIComponent(interviewId)}`, { method: "DELETE" }), [queryKeys.interviews])
      .then(() => {
        if (currentInterview?.interview_id === interviewId) {
          onSelect({ resume_run_id: item.resume_run_id || "", interview_id: "" });
          setAnswer("");
          setVoiceQuestion("");
        }
        notify("面试记录已删除");
      })
      .catch((error) => notify(`面试记录删除失败：${error.message || "未知错误"}`))
      .finally(() => setDeletingInterviewId(""));
  };
  const selectSession = (item) => {
    selectMode(interviewMode(item));
    onSelect(item);
  };
  const serverQuestion = pendingQuestion?.text || "";
  useEffect(() => setVoiceQuestion(""), [currentInterview?.interview_id, serverQuestion]);
  const question = voiceQuestion || serverQuestion || (nextQuestionError || (waitingForQuestion ? "正在同步下一题..." : `选择${modeMeta.label}后开始面试，问题会绑定当前岗位会话。`));
  const serverQuestionObject = pendingQuestion;
  const blockLabels = (serverQuestionObject?.knowledge_block_refs || []).map((item) => item.label).filter(Boolean);
  const currentSessions = sessions
    .filter((item) => interviewMode(item) === mode)
    .slice(-20)
    .reverse();
  return <section className="sa-interview-layout"><div className="sa-card sa-interview-main"><div className="sa-interview-mode-switch" role="tablist" aria-label="面试模式"><span className="sa-mode-switch-label">面试模式</span>{INTERVIEW_MODES.map((item) => <button key={item.value} type="button" role="tab" aria-selected={mode === item.value} className={`sa-mode-option ${mode === item.value ? "active" : ""}`} onClick={() => selectMode(item.value)}><strong>{item.label}</strong><small>{item.description}</small></button>)}</div><div className="sa-card-head"><div><h2>{modeMeta.label}</h2><p>{context.jd?.role_title || "尚未选择岗位"} · {currentDisplayStatus}</p></div><span className={`sa-badge ${currentIsLive ? "green" : "orange"}`}>{currentDisplayStatus}</span></div><div className="sa-question"><span>当前问题 · {modeMeta.label}{blockLabels.length ? ` · ${blockLabels.join(" + ")}` : ""}</span><strong>{question}</strong></div>{currentIsLive && <VoiceInterviewControls interview={currentInterview} question={question} setAnswer={setAnswer} onQuestion={setVoiceQuestion} />}<textarea rows={8} value={answer} readOnly={Boolean(currentInterview && !currentIsLive)} onChange={(event) => setAnswer(event.target.value)} placeholder="输入回答，或使用上方语音面试。" />{waitingForQuestion && nextQuestionError && <div className="sa-actions"><button className="sa-button primary" onClick={retryNextQuestion} disabled={retryingNextQuestion}><RefreshCw size={15} className={retryingNextQuestion ? "spin" : ""} />{retryingNextQuestion ? "正在重试..." : "重试生成下一题"}</button></div>}{(!currentInterview || currentIsLive) && <div className="sa-actions"><button className="sa-button primary" onClick={currentIsLive ? submit : start}>{currentIsLive ? <><Send size={15} />提交回答</> : <><MessageCircle size={15} />开始{modeMeta.label}</>}</button></div>}</div><div className="sa-card"><div className="sa-card-head"><div><h2>会话记录</h2><p>按当前模式显示当前岗位关联的面试。</p></div><span className="sa-badge">{currentSessions.length} 条</span></div><div className="sa-session-list">{currentSessions.map((item) => <div className="sa-session-row" key={item.interview_id}><button type="button" className={currentInterview?.interview_id === item.interview_id ? "active" : ""} onClick={() => selectSession(item)} disabled={deletingInterviewId === item.interview_id}><strong>{interviewMode(item) === "gap_analysis" ? "查漏补缺" : "真实模拟"}</strong><span>{item.profile?.role_title || item.role_title || context.jd?.role_title || "当前岗位"}</span><small>{interviewDisplayStatus(item, sessions)} · {formatTime(item.updated_at)}</small></button><button type="button" className="sa-icon-button" title="删除面试记录" aria-label={`删除面试记录 ${interviewMode(item) === "gap_analysis" ? "查漏补缺" : "真实模拟"}`} onClick={() => deleteInterview(item)} disabled={Boolean(deletingInterviewId)}><Trash2 size={15} /></button></div>)}{!currentSessions.length && <EmptyState text={`暂无${modeMeta.label}会话。`} />}</div>{currentInterview && <InterviewHistoryDetail session={currentInterview} />} </div></section>;
}

function InterviewHistoryDetail({ session }) {
  const turns = Array.isArray(session?.turns) ? session.turns : [];
  const pending = session?.pending_question;
  return <div className="sa-interview-history-detail"><div className="sa-section-title"><strong>问答回放</strong><span>{turns.length} 轮</span></div>{turns.map((turn, index) => <article className="sa-interview-turn" key={turn.question?.question_id || `${session.interview_id}-${index}`}><div><span>面试官 · 第 {index + 1} 题</span><p>{turn.question?.text || "未记录问题"}</p></div><div><span>候选人回答</span><p>{turn.answer || "未记录回答"}</p></div></article>)}{pending && <article className="sa-interview-turn pending"><div><span>当前待回答问题</span><p>{pending.text || "未记录问题"}</p></div></article>}{!turns.length && !pending && <EmptyState text="该面试暂无问答记录。" />}</div>;
}

function VoiceInterviewControls({ interview, question, setAnswer, onQuestion }) {
  const [voice, setVoice] = useState({ phase: "idle", detail: "未连接麦克风", transcript: "", provider: "", error: "" });
  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const silenceMonitorRef = useRef(0);
  const transcriptSilenceTimerRef = useRef(0);
  const silenceSinceRef = useRef(0);
  const speechDetectedRef = useRef(false);
  const streamIdRef = useRef("");
  const startedAtRef = useRef(0);
  const transcriptRef = useRef("");
  const providerRef = useRef("");
  const phaseRef = useRef("idle");
  const shouldListenRef = useRef(false);
  const stoppingRef = useRef(false);
  const finalizingRef = useRef(false);
  const sendChainRef = useRef(Promise.resolve());

  const updateVoice = (next) => setVoice((current) => {
    const value = { ...current, ...next };
    phaseRef.current = value.phase;
    return value;
  });
  const clearSilenceDetection = () => {
    if (silenceMonitorRef.current) {
      window.clearInterval(silenceMonitorRef.current);
      silenceMonitorRef.current = 0;
    }
    if (transcriptSilenceTimerRef.current) {
      window.clearTimeout(transcriptSilenceTimerRef.current);
      transcriptSilenceTimerRef.current = 0;
    }
    silenceSinceRef.current = 0;
    speechDetectedRef.current = false;
    analyserRef.current = null;
    const context = audioContextRef.current;
    audioContextRef.current = null;
    if (context) context.close().catch(() => {});
  };
  const scheduleTranscriptSilenceFallback = () => {
    if (transcriptSilenceTimerRef.current) window.clearTimeout(transcriptSilenceTimerRef.current);
    if (!shouldListenRef.current || finalizingRef.current || !transcriptRef.current.trim()) return;
    transcriptSilenceTimerRef.current = window.setTimeout(() => {
      transcriptSilenceTimerRef.current = 0;
      if (shouldListenRef.current && !finalizingRef.current && transcriptRef.current.trim()) {
        updateVoice({ detail: "检测到停顿，正在自动提交回答" });
        finalizeTurn();
      }
    }, AUTO_FINALIZE_SILENCE_MS);
  };
  const startSilenceMonitor = (stream) => {
    clearSilenceDetection();
    const AudioContextConstructor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextConstructor || !stream) return;
    try {
      const audioContext = new AudioContextConstructor();
      const analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(stream);
      analyser.fftSize = 2048;
      source.connect(analyser);
      audioContext.resume?.().catch(() => {});
      const samples = new Uint8Array(analyser.fftSize);
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      silenceMonitorRef.current = window.setInterval(() => {
        if (!shouldListenRef.current || finalizingRef.current || !analyserRef.current) return;
        analyser.getByteTimeDomainData(samples);
        let squareSum = 0;
        for (let index = 0; index < samples.length; index += 1) {
          const amplitude = (samples[index] - 128) / 128;
          squareSum += amplitude * amplitude;
        }
        const rms = Math.sqrt(squareSum / samples.length);
        if (rms >= AUDIO_SILENCE_RMS_THRESHOLD) {
          speechDetectedRef.current = true;
          silenceSinceRef.current = 0;
          return;
        }
        if (!speechDetectedRef.current || !transcriptRef.current.trim()) return;
        if (!silenceSinceRef.current) silenceSinceRef.current = performance.now();
        if (performance.now() - silenceSinceRef.current >= AUTO_FINALIZE_SILENCE_MS) {
          updateVoice({ detail: "检测到停顿，正在自动提交回答" });
          finalizeTurn();
        }
      }, 100);
    } catch (_) {
      clearSilenceDetection();
    }
  };
  const clearAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }
    window.speechSynthesis?.cancel();
  };
  const releaseMedia = () => {
    clearSilenceDetection();
    try { recorderRef.current?.stop(); } catch (_) {}
    recorderRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    try { recognitionRef.current?.abort(); } catch (_) {}
    recognitionRef.current = null;
  };
  const closeSocket = () => {
    const socket = socketRef.current;
    socketRef.current = null;
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "close" }));
    try { socket?.close(); } catch (_) {}
  };
  const resetVoice = () => {
    shouldListenRef.current = false;
    stoppingRef.current = false;
    finalizingRef.current = false;
    clearAudio();
    releaseMedia();
    closeSocket();
    streamIdRef.current = "";
    transcriptRef.current = "";
    providerRef.current = "";
    updateVoice({ phase: "idle", detail: "麦克风未启动", transcript: "", provider: "", error: "" });
  };
  useEffect(() => resetVoice, [interview.interview_id]);

 const socketUrl = () => {
   const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    const token = getAuthToken();
    const query = token ? "?token=" + encodeURIComponent(token) : "";
    return scheme + "//" + window.location.host + "/ws/career/interviews/" + encodeURIComponent(interview.interview_id) + "/voice" + query;
 };
  const blobToBase64 = (blob) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || "").split(",", 2)[1] || "");
    reader.onerror = () => reject(reader.error || new Error("音频编码失败"));
    reader.readAsDataURL(blob);
  });
  const sendAudio = async (blob, mediaType, extra = {}) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN || !streamIdRef.current || !blob?.size) return;
    const audio_base64 = await blobToBase64(blob);
    socket.send(JSON.stringify({
      type: "audio",
      stream_id: streamIdRef.current,
      audio_base64,
      media_type: mediaType,
      elapsed_ms: Math.max(0, Math.round(performance.now() - startedAtRef.current)),
      auto_finalize: false,
      ...extra,
    }));
  };
  const queueAudio = (blob, mediaType, extra = {}) => {
    sendChainRef.current = sendChainRef.current
      .then(() => sendAudio(blob, mediaType, extra))
      .catch((error) => updateVoice({ phase: "error", detail: error.message || "音频发送失败", error: error.message || "" }));
    return sendChainRef.current;
  };
  const completeTurn = () => {
    const socket = socketRef.current;
    if (!finalizingRef.current || stoppingRef.current || !socket || socket.readyState !== WebSocket.OPEN || !streamIdRef.current) return;
    sendChainRef.current.finally(() => {
      if (stoppingRef.current || socket.readyState !== WebSocket.OPEN || !streamIdRef.current) return;
      updateVoice({ phase: "processing", detail: "正在提交语音回答" });
      socket.send(JSON.stringify({ type: "stop", stream_id: streamIdRef.current }));
    });
  };
  const playback = (payload, after) => {
    const text = String(payload?.text || "").trim();
    const audioBase64 = String(payload?.audio_base64 || "").trim();
    const finish = () => {
      if (shouldListenRef.current && !stoppingRef.current) after?.();
    };
    updateVoice({ phase: "speaking", detail: "面试官正在提问" });
    if (audioBase64) {
      const source = `data:${payload.media_type || "audio/mpeg"};base64,${audioBase64}`;
      const audio = new Audio(source);
      audioRef.current = audio;
      audio.onended = finish;
      audio.onerror = () => {
        audioRef.current = null;
        playBrowserSpeech(text, finish);
      };
      audio.play().catch(() => playBrowserSpeech(text, finish));
      return;
    }
    playBrowserSpeech(text, finish);
  };
  const playBrowserSpeech = (text, after) => {
    if (!text || !window.speechSynthesis) {
      after?.();
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "zh-CN";
    utterance.rate = 1;
    utterance.onend = after;
    utterance.onerror = after;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };
  const finalizeTurn = () => {
    const socket = socketRef.current;
    if (finalizingRef.current || !socket || socket.readyState !== WebSocket.OPEN || !streamIdRef.current) return;
    finalizingRef.current = true;
    clearSilenceDetection();
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      try { recorder.requestData(); recorder.stop(); } catch (_) { completeTurn(); }
      return;
    }
    const recognition = recognitionRef.current;
    if (recognition) {
      try { recognition.stop(); } catch (_) { completeTurn(); }
      return;
    }
    completeTurn();
  };
  const sendBrowserTranscript = async (text) => {
    const normalized = String(text || "").trim();
    if (!normalized) return;
    speechDetectedRef.current = true;
    transcriptRef.current = normalized;
    setAnswer(normalized);
    updateVoice({ phase: "listening", detail: "正在识别你的回答", transcript: normalized });
    await queueAudio(new Blob([normalized], { type: "text/plain;charset=utf-8" }), "text/plain", { auto_finalize: false });
    scheduleTranscriptSilenceFallback();
  };
  const startBrowserRecognition = () => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      updateVoice({ phase: "error", detail: "浏览器不支持语音识别", error: "请使用 Chrome 或 Edge，或配置后端 ASR。" });
      return;
    }
    const recognition = new Recognition();
    recognitionRef.current = recognition;
    const accumulatedText = transcriptRef.current.trim();
    let finalText = "";
    recognition.lang = "zh-CN";
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => updateVoice({ phase: "listening", detail: "请开始回答" });
    recognition.onresult = (event) => {
      let interim = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const text = event.results[index][0]?.transcript || "";
        if (event.results[index].isFinal) finalText += text;
        else interim += text;
      }
      sendBrowserTranscript(`${accumulatedText} ${finalText} ${interim}`.trim()).catch((error) => updateVoice({ phase: "error", detail: error.message || "转写发送失败", error: error.message || "" }));
    };
    recognition.onspeechend = scheduleTranscriptSilenceFallback;
    recognition.onerror = (event) => {
      if (event.error === "aborted" || stoppingRef.current) return;
      if (["not-allowed", "audio-capture", "network"].includes(event.error)) shouldListenRef.current = false;
      const message = { "not-allowed": "麦克风权限被拒绝", "audio-capture": "未检测到麦克风", network: "浏览器语音识别服务不可用" }[event.error] || `语音识别失败：${event.error}`;
      updateVoice({ phase: "error", detail: message, error: message });
    };
    recognition.onend = () => {
      if (finalizingRef.current) {
        completeTurn();
        return;
      }
      if (shouldListenRef.current && !finalizingRef.current && !stoppingRef.current) {
        window.setTimeout(() => {
          if (shouldListenRef.current && !finalizingRef.current) startBrowserRecognition();
        }, 300);
      }
    };
    recognition.start();
  };
  const startMicrophoneStreaming = async (provider) => {
    const micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    streamRef.current = micStream;
    startSilenceMonitor(micStream);
    if (provider === "buffered-text") {
      startBrowserRecognition();
      return;
    }
    const formats = ["audio/webm;codecs=opus", "audio/webm"].filter((type) => MediaRecorder.isTypeSupported?.(type));
    const recorder = new MediaRecorder(micStream, formats[0] ? { mimeType: formats[0] } : undefined);
    recorderRef.current = recorder;
    recorder.onstart = () => updateVoice({ phase: "listening", detail: "正在聆听你的回答" });
    recorder.ondataavailable = (event) => {
      if (event.data?.size) queueAudio(event.data, recorder.mimeType || "audio/webm");
    };
    recorder.onstop = () => {
      if (finalizingRef.current) completeTurn();
    };
    recorder.onerror = () => updateVoice({ phase: "error", detail: "麦克风录音失败", error: "MediaRecorder error" });
    recorder.start(700);
  };
  const applyVoiceResponse = (payload) => {
    const events = Array.isArray(payload?.events) ? payload.events : [];
    const latest = events.at(-1)?.text;
    if (latest) {
      transcriptRef.current = latest;
      setAnswer(latest);
      updateVoice({ transcript: latest, detail: payload.accepted_final ? "已识别完整回答" : "正在识别你的回答" });
    }
    const nextQuestion = payload?.next_question;
    if (nextQuestion?.text) {
      transcriptRef.current = "";
      setAnswer("");
      onQuestion(nextQuestion.text);
      finalizingRef.current = false;
      releaseMedia();
      updateVoice({ phase: "speaking", detail: "已生成下一题", transcript: "" });
      playback(payload.playback || { text: nextQuestion.text }, () => startMicrophoneStreaming(providerRef.current));
    } else if (payload?.answer_result?.status === "completed") {
      updateVoice({ phase: "finished", detail: "本次面试已完成" });
      shouldListenRef.current = false;
      releaseMedia();
      closeSocket();
    } else if (payload?.submitted) {
      finalizingRef.current = false;
      updateVoice({ phase: "listening", detail: "继续回答，或点击结束本轮" });
    }
  };
  const startVoice = () => {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      updateVoice({ phase: "error", detail: "当前浏览器不支持麦克风录音", error: "Media devices unavailable" });
      return;
    }
    resetVoice();
    shouldListenRef.current = true;
    startedAtRef.current = performance.now();
    updateVoice({ phase: "connecting", detail: "正在连接语音面试" });
    const socket = new WebSocket(socketUrl());
    socketRef.current = socket;
    socket.onopen = () => socket.send(JSON.stringify({ type: "start", stream_id: `browser-${crypto.randomUUID?.() || Date.now()}`, media_type: "audio/webm" }));
    socket.onmessage = (event) => {
      let payload;
      try { payload = JSON.parse(event.data); } catch (_) { return; }
      if (payload.type === "error") {
        updateVoice({ phase: "error", detail: payload.content || "语音服务错误", error: payload.content || "" });
        return;
      }
      if (payload.type === "started") {
        streamIdRef.current = payload.stream_id;
        const provider = payload.asr || "";
        providerRef.current = provider;
        updateVoice({ phase: "speaking", detail: provider === "buffered-text" ? "使用浏览器语音识别" : "语音会话已连接", provider, error: "" });
        playback(payload.playback || { text: payload.question?.text || question }, () => startMicrophoneStreaming(provider));
        return;
      }
      if (payload.type === "speech_events" || payload.type === "stopped") applyVoiceResponse(payload);
    };
    socket.onerror = () => updateVoice({ phase: "error", detail: "无法连接语音服务", error: "WebSocket connection failed" });
    socket.onclose = () => {
      if (shouldListenRef.current && phaseRef.current !== "finished" && phaseRef.current !== "error") updateVoice({ phase: "error", detail: "语音连接已关闭", error: "Voice socket closed" });
    };
  };
  const stopVoice = () => {
    stoppingRef.current = true;
    shouldListenRef.current = false;
    const socket = socketRef.current;
    const streamId = streamIdRef.current;
    clearAudio();
    releaseMedia();
    if (socket && socket.readyState === WebSocket.OPEN && streamId) socket.send(JSON.stringify({ type: "interrupt", stream_id: streamId }));
    window.setTimeout(() => {
      closeSocket();
      streamIdRef.current = "";
      updateVoice({ phase: "idle", detail: "语音面试已停止", transcript: "" });
    }, 120);
  };
  const isActive = ["connecting", "speaking", "listening", "processing"].includes(voice.phase);
  const voiceClass = voice.phase === "listening" ? "recording" : voice.phase === "error" ? "error" : "";
  return <div className={`sa-voice-panel ${voiceClass}`}><div className="sa-voice-head"><div><strong>语音面试</strong><small>麦克风音频通过当前面试会话发送；未配置后端 ASR 时自动使用浏览器转写，回答停顿约 4 秒会自动提交。</small></div><span className="sa-voice-pill">{voice.phase === "listening" ? "正在聆听" : voice.phase === "speaking" ? "正在提问" : voice.phase === "processing" ? "处理中" : voice.phase === "error" ? "不可用" : "未启动"}</span></div><div className="sa-voice-controls">{!isActive && voice.phase !== "finished" && <button type="button" className="sa-button primary" onClick={startVoice}><Mic size={15} />开始语音面试</button>}{voice.phase === "listening" && <button type="button" className="sa-button" onClick={finalizeTurn}><Send size={15} />结束本轮</button>}{isActive && <button type="button" className="sa-button ghost" onClick={stopVoice}><MicOff size={15} />停止语音</button>}{voice.phase === "finished" && <span className="sa-inline-note">面试报告已生成。</span>}</div><div className="sa-voice-meta"><span><Radio size={12} /> {voice.detail}</span>{voice.provider && <span>ASR：{voice.provider}</span>}</div><div className="sa-voice-transcript"><span>实时转写</span><p>{voice.transcript || "开始后这里会显示你的语音转写。"}</p></div></div>;
}

function AdminPanel({ notify }) {
  const queryClient = useQueryClient();
  const flowQuery = useCareerQuery(queryKeys.adminFlow, "/api/admin/agent-flow?limit=500", true, {
    refetchInterval: 5000,
    refetchOnWindowFocus: true,
  });
  const [streamEvents, setStreamEvents] = useState([]);
  const [selectedAgentType, setSelectedAgentType] = useState("smart");
  const [selectedCallId, setSelectedCallId] = useState("");
  const [selectedStepId, setSelectedStepId] = useState("");
  const [liveStatus, setLiveStatus] = useState("连接中");

  useEffect(() => {
    if (typeof EventSource === "undefined") {
      setLiveStatus("实时流不可用");
      return undefined;
    }
    const source = new EventSource("/api/admin/agent-events/stream");
    const handleEvent = (event) => {
      try {
        const item = normalizeAdminEvent(JSON.parse(event.data));
        setStreamEvents((current) => {
          const seen = new Set();
          return [item, ...current]
            .filter((row) => {
              if (seen.has(row.id)) return false;
              seen.add(row.id);
              return true;
            })
            .slice(0, 500);
        });
      } catch (_) {
        setLiveStatus("事件解析失败");
      }
    };
    source.addEventListener("agent", handleEvent);
    source.onopen = () => setLiveStatus("实时连接");
    source.onerror = () => setLiveStatus("实时断开，历史可读");
    return () => source.close();
  }, []);

  const events = useMemo(() => {
    const merged = new Map();
    [...(flowQuery.data?.events || []), ...streamEvents].map(normalizeAdminEvent).forEach((event) => merged.set(event.id, event));
    return [...merged.values()].sort((a, b) => adminNumber(b.created_at) - adminNumber(a.created_at));
  }, [flowQuery.data?.events, streamEvents]);
  const metrics = flowQuery.data?.summary || {
    career_runs: events.filter((item) => adminAgentType(item) === "career").length,
    project_analysis_runs: events.filter((item) => adminAgentType(item) === "project_analyst").length,
    interview_questions: events.filter((item) => adminAgentType(item) === "interviewer").length,
    failed: events.filter((item) => adminStatusClass(item.status) === "failed").length,
    tool_calls: events.reduce((sum, item) => sum + adminNumber(item.tool_call_count), 0),
    tokens: events.reduce((sum, item) => sum + adminNumber(item.token_count), 0),
  };
  const selectedAgent = ADMIN_AGENTS.find((item) => item.id === selectedAgentType) || ADMIN_AGENTS[0];
  const agentRows = ADMIN_AGENTS.map((agent) => {
    const calls = adminCallsFor(events, agent.id);
    return { ...agent, calls, status: adminAggregateStatus(calls.flatMap((call) => call.events)) };
  });
  const calls = adminCallsFor(events, selectedAgent.id);
  const selectedCall = calls.find((call) => call.id === selectedCallId) || calls[0] || null;
  const steps = adminStepsFor(selectedCall);
  const selectedStep = steps.find((step) => step.id === selectedStepId) || steps[0] || null;

  useEffect(() => {
    if (!calls.some((call) => call.id === selectedCallId)) setSelectedCallId(calls[0]?.id || "");
  }, [calls, selectedCallId]);

  useEffect(() => {
    if (!steps.some((step) => step.id === selectedStepId)) setSelectedStepId(steps[0]?.id || "");
  }, [steps, selectedStepId]);

  const refreshAdmin = () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.adminFlow });
    notify?.("运行后台已刷新");
  };
  const metricRows = [
    ["career_runs", "Career 运行"],
    ["project_analysis_runs", "项目分析"],
    ["interview_questions", "面试题"],
    ["failed", "失败"],
    ["tool_calls", "工具调用"],
    ["tokens", "Token"],
  ];

  return <section className="sa-admin-layout">
    <div className="sa-card sa-wide">
      <div className="sa-card-head"><div><h2>Agent 运行后台</h2><p>只读查看主 Agent、Career Agent、简历生成、核验、面试和项目分析的调用链。</p></div><div className="sa-actions compact"><span className={`sa-badge ${liveStatus === "实时连接" ? "green" : "orange"}`}>{liveStatus}</span><button type="button" className="sa-button" onClick={refreshAdmin}><RefreshCw size={15} />刷新</button></div></div>
      {flowQuery.isError && <div className="sa-admin-error">{flowQuery.error?.message || "运行后台读取失败"}</div>}
      <div className="sa-admin-metrics">{metricRows.map(([key, label]) => <div key={key}><strong>{adminNumber(metrics[key]).toLocaleString()}</strong><span>{label}</span></div>)}</div>
    </div>
    <div className="sa-card sa-admin-card"><div className="sa-card-head"><div><h2>Agent 列表</h2><p>固定展示 8 个 Agent。</p></div></div><div className="sa-admin-list">{agentRows.map((agent) => <button type="button" key={agent.id} className={`sa-admin-row ${selectedAgent.id === agent.id ? "active" : ""}`} onClick={() => { setSelectedAgentType(agent.id); setSelectedCallId(""); setSelectedStepId(""); }}><span className={`sa-admin-dot ${adminStatusClass(agent.status)}`} /><div><strong>{agent.name}</strong><small>{agent.calls.length} 次调用</small></div><em>{agent.status}</em></button>)}</div></div>
    <div className="sa-card sa-admin-card"><div className="sa-card-head"><div><h2>调用流程</h2><p>{selectedAgent.name}</p></div><span className="sa-badge">{calls.length} 次</span></div><div className="sa-admin-list">{calls.length ? calls.map((call) => <button type="button" key={call.id} className={`sa-admin-row ${selectedCall?.id === call.id ? "active" : ""}`} onClick={() => { setSelectedCallId(call.id); setSelectedStepId(""); }}><span className={`sa-admin-dot ${adminStatusClass(call.status)}`} /><div><strong>{call.title}</strong><small>{formatTime(call.created_at)} · {adminDuration(call.elapsed_ms)}</small></div><em>{call.status}</em></button>) : <EmptyState text="该 Agent 暂无调用记录。" />}</div></div>
    <div className="sa-card sa-admin-card sa-admin-detail"><div className="sa-card-head"><div><h2>完整输入 / 输出</h2><p>{selectedCall ? selectedCall.title : "选择一次调用后查看详情"}</p></div><span className="sa-badge">{steps.length} 步</span></div>{selectedStep ? <><div className="sa-admin-steps">{steps.map((step) => <button type="button" key={step.id} className={selectedStep.id === step.id ? "active" : ""} onClick={() => setSelectedStepId(step.id)}><span>{String(step.index).padStart(2, "0")}</span><strong>{step.title}</strong><small>{step.kind} · {adminDuration(step.elapsed_ms)}</small></button>)}</div><div className="sa-admin-json-grid"><div><strong>输入</strong><pre>{adminRedactedText(selectedStep.input) || "无输入记录"}</pre></div><div><strong>输出 / 结果</strong><pre>{adminRedactedText(selectedStep.output) || "无输出记录"}</pre></div><div className="sa-admin-raw"><strong>原始步骤记录</strong><pre>{adminRedactedText(selectedStep.raw)}</pre></div></div></> : <EmptyState text={flowQuery.isLoading ? "正在读取运行记录..." : "暂无可展示的调用详情。"} />}</div>
  </section>;
}

function LegacyCodeAgentPanel({ projects = [], projectsLoading = false, notify, setTask }) {
  const [projectId, setProjectId] = useState("");
  const [path, setPath] = useState(".");
  const [files, setFiles] = useState([]);
  const [selectedPath, setSelectedPath] = useState("");
  const [content, setContent] = useState("");
  const [sha256, setSha256] = useState("");
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const socketRef = useRef(null);
  const normalized = useMemo(() => projects.map((item) => item?.project_source || item).filter((item) => item?.project_id), [projects]);
  useEffect(() => { if (!projectId && normalized[0]) setProjectId(normalized[0].project_id); }, [normalized, projectId]);
  const loadFiles = async (nextPath = ".") => {
    if (!projectId) return;
    setLoading(true);
    try { const data = await api(`/api/files?project_id=${encodeURIComponent(projectId)}&path=${encodeURIComponent(nextPath)}`); setFiles(data.items || []); setPath(nextPath); }
    catch (error) { notify?.(error.message || "读取文件列表失败"); }
    finally { setLoading(false); }
  };
  useEffect(() => { setPath("."); setFiles([]); setSelectedPath(""); setContent(""); setMessages([]); if (projectId) loadFiles("."); }, [projectId]);
  const openFile = async (file) => {
    try { const data = await api(`/api/file?project_id=${encodeURIComponent(projectId)}&path=${encodeURIComponent(file)}`); setSelectedPath(file); setContent(data.content || ""); setSha256(data.sha256 || ""); }
    catch (error) { notify?.(error.message || "读取文件失败"); }
  };
  const save = async () => {
    if (!selectedPath || busy) return;
    setBusy(true); setTask?.({ state: "running", text: "正在保存文件..." });
    try { const data = await api("/api/save", { method: "POST", body: jsonBody({ project_id: projectId, path: selectedPath, content, ...(sha256 ? { expected_sha256: sha256 } : {}) }) }); setSha256(data.sha256 || ""); setTask?.({ state: "done", text: "文件已保存" }); notify?.("文件已保存"); }
    catch (error) { setTask?.({ state: "error", text: error.message || "保存失败" }); notify?.(error.status === 409 ? "文件已变化，请重新读取" : (error.message || "保存失败")); }
    finally { setBusy(false); }
  };
  const send = () => {
    const text = prompt.trim(); if (!text || !projectId || busy) return;
    const history = messages.slice(-6).map((item) => `${item.role}: ${item.content}`).join("\n");
    const message = `${selectedPath ? `Selected file: ${selectedPath}\n` : ""}${history ? `Conversation:\n${history}\n` : ""}Request: ${text}`;
    setMessages((current) => [...current, { role: "user", content: text }, { role: "assistant", content: "", pending: true }]); setPrompt(""); setBusy(true); setTask?.({ state: "running", text: "Code Agent 处理中..." });
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/chat?token=${encodeURIComponent(getAuthToken())}`);
    socket.onopen = () => socket.send(JSON.stringify({ message, original_message: text, domain: "code", project_id: projectId, token: getAuthToken(), session_id: `code-${Date.now()}` }));
    socket.onmessage = (event) => { let data; try { data = JSON.parse(event.data); } catch { return; } if (data.type === "status") setTask?.({ state: "running", text: data.content || "Code Agent 处理中..." }); if (data.type === "delta") setMessages((current) => { const next = [...current]; const last = next.at(-1); if (last?.role === "assistant") next[next.length - 1] = { ...last, content: last.content + (data.content || "") }; return next; }); if (data.type === "result" || data.type === "error") { setMessages((current) => { const next = [...current]; const last = next.at(-1); if (last?.role === "assistant") next[next.length - 1] = { ...last, content: data.content || "无回复", pending: false, error: data.type === "error", tools: data.tool_calls || [] }; return next; }); setBusy(false); setTask?.({ state: data.type === "error" ? "error" : "done", text: data.content || (data.type === "error" ? "Code Agent 失败" : "完成") }); if (data.type === "result") { loadFiles(path); if (selectedPath) openFile(selectedPath); } } };
    socket.onerror = () => { setBusy(false); setTask?.({ state: "error", text: "Code Agent 连接失败" }); };
  };
  const parent = path === "." ? null : (path.split("/").slice(0, -1).join("/") || ".");
  useEffect(() => () => { if (socketRef.current && socketRef.current.readyState < 2) socketRef.current.close(); }, []);
  return <section className="sa-code-workspace"><div className="sa-code-toolbar"><label>项目<select value={projectId} onChange={(event) => setProjectId(event.target.value)} disabled={projectsLoading || !normalized.length}><option value="">{projectsLoading ? "读取中..." : "选择已导入项目"}</option>{normalized.map((item) => <option value={item.project_id} key={item.project_id}>{item.display_name || item.project_id}</option>)}</select></label><span className="sa-code-project-meta">{projectId ? "仅可访问这个已登记项目" : ""}</span></div>{!projectId ? <EmptyPanel icon={<Folder size={28} />} title="选择项目" text="Code Agent 只在已导入项目内工作。" /> : <div className="sa-code-grid"><div className="sa-card sa-code-files"><div className="sa-card-head"><div><h2>项目文件</h2><p>{path}</p></div><button type="button" className="sa-icon-button" onClick={() => loadFiles(path)} aria-label="刷新"><RefreshCw size={15} className={loading ? "spin" : ""} /></button></div>{parent && <button type="button" className="sa-code-file-row" onClick={() => loadFiles(parent)}><FolderOpen size={15} />..</button>}{files.map((item) => { const full = path === "." ? item.name : `${path}/${item.name}`; return <button type="button" key={full} className={`sa-code-file-row ${selectedPath === full ? "active" : ""}`} onClick={() => item.type === "dir" ? loadFiles(full) : openFile(full)}>{item.type === "dir" ? <Folder size={15} /> : <FileText size={15} />}<span>{item.name}</span></button>; })}{!files.length && !loading && <EmptyState text="目录为空" />}</div><div className="sa-code-editor-chat"><div className="sa-card sa-code-editor"><div className="sa-card-head"><div><h2>{selectedPath || "文件编辑器"}</h2><p>已启用乐观锁</p></div><button type="button" className="sa-button primary" onClick={save} disabled={!selectedPath || busy}>保存</button></div><textarea value={content} onChange={(event) => setContent(event.target.value)} disabled={!selectedPath} spellCheck="false" placeholder="选择一个文件" /></div><div className="sa-card sa-code-chat"><div className="sa-card-head"><div><h2>Code Agent</h2><p>读取、修改、测试和解释当前项目</p></div><span className={`sa-badge ${busy ? "orange" : "green"}`}>{busy ? "处理中" : "就绪"}</span></div><div className="sa-code-messages">{messages.map((item, index) => <article className={`sa-code-message ${item.role} ${item.error ? "error" : ""}`} key={index}><span>{item.role === "user" ? "你" : "Code Agent"}</span><p>{item.content || (item.pending ? "正在处理..." : "")}</p>{item.tools?.length ? <small>工具：{item.tools.map((tool) => tool.tool).filter(Boolean).join("、")}</small> : null}</article>)}{!messages.length && <EmptyState text="描述要完成的代码任务" />}</div><div className="sa-code-composer"><textarea rows={3} value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); send(); } }} placeholder="例如：检查项目并运行测试" disabled={busy} /><button type="button" className="sa-button primary" onClick={send} disabled={!prompt.trim() || busy}><Send size={15} />发送</button></div></div></div></div>}</section>;
}

const CODE_AGENT_SESSIONS_KEY = "smart-harness-code-agent-sessions-v1";
const CODE_AGENT_SETTINGS_KEY = "smart-harness-code-agent-settings-v1";

function readJsonStorage(key, fallback) {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(key) || "");
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
}

function writeJsonStorage(key, value) {
  try { window.localStorage.setItem(key, JSON.stringify(value)); } catch { /* browser storage can be unavailable */ }
}

function readCodeAgentSettings() {
  const saved = readJsonStorage(CODE_AGENT_SETTINGS_KEY, {});
  const baseUrl = String(saved?.baseUrl || "").trim();
  return {
    model: String(saved?.model || ""),
    baseUrl: baseUrl && /^https?:\/\//i.test(baseUrl) ? baseUrl : "",
    // API keys are intentionally never restored from browser storage.
    apiKey: "",
    reasoning: ["low", "medium", "high", "max"].includes(saved?.reasoning) ? saved.reasoning : "medium",
    agentPrompt: String(saved?.agentPrompt || ""),
  };
}

function joinProjectPath(base, name) {
  return base && base !== "." ? `${base}/${name}` : name;
}

function codeSessionTitle(text) {
  const value = String(text || "").trim().replace(/\s+/g, " ");
  return value ? value.slice(0, 32) : "新会话";
}

function CodeAgentPanel({ projects = [], projectsLoading = false, notify, setTask }) {
  const [projectId, setProjectId] = useState(() => window.localStorage.getItem("smart-harness-code-project-id") || "");
  const [tree, setTree] = useState({});
  const [expandedDirs, setExpandedDirs] = useState(() => new Set(["."]));
  const [loadingDirs, setLoadingDirs] = useState(() => new Set());
  const [activePath, setActivePath] = useState("");
  const [activeContent, setActiveContent] = useState("");
  const [activeHash, setActiveHash] = useState("");
  const [fileBusy, setFileBusy] = useState(false);
  const [messageDraft, setMessageDraft] = useState("");
  const [sessions, setSessions] = useState(() => readJsonStorage(CODE_AGENT_SESSIONS_KEY, []));
  const [activeSessionId, setActiveSessionId] = useState(() => window.localStorage.getItem("smart-harness-code-session-id") || "");
  const [agentBusy, setAgentBusy] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState(readCodeAgentSettings);
  const socketRef = useRef(null);
  const runningSessionIdRef = useRef("");
  const pauseRequestedRef = useRef(false);
  const normalizedProjects = useMemo(() => projects.map((item) => item?.project_source || item).filter((item) => item?.project_id), [projects]);
  const selectedProject = normalizedProjects.find((item) => item.project_id === projectId) || null;
  const projectSessions = useMemo(() => sessions.filter((item) => item.project_id === projectId), [sessions, projectId]);
  const activeSession = projectSessions.find((item) => item.id === activeSessionId) || projectSessions[0] || null;
  const activeMessages = activeSession?.messages || [];

  const persistSessions = (updater) => {
    setSessions((current) => {
      const next = typeof updater === "function" ? updater(current) : updater;
      writeJsonStorage(CODE_AGENT_SESSIONS_KEY, next);
      return next;
    });
  };
  const createSession = (title = "新会话") => {
    if (!projectId) return "";
    const session = { id: `code-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`, project_id: projectId, title, messages: [], created_at: Date.now(), updated_at: Date.now() };
    persistSessions((current) => [session, ...current]);
    setActiveSessionId(session.id);
    window.localStorage.setItem("smart-harness-code-session-id", session.id);
    return session.id;
  };
  const updateSessionMessages = (sessionId, updater) => {
    persistSessions((current) => current.map((item) => {
      if (item.id !== sessionId) return item;
      const messages = typeof updater === "function" ? updater(item.messages || []) : updater;
      const firstUser = messages.find((message) => message.role === "user")?.content || item.title;
      return { ...item, title: codeSessionTitle(firstUser), messages, updated_at: Date.now() };
    }));
  };

  const loadDirectory = async (dir = ".") => {
    if (!projectId || loadingDirs.has(dir)) return;
    setLoadingDirs((current) => new Set([...current, dir]));
    try {
      const data = await api(`/api/files?project_id=${encodeURIComponent(projectId)}&path=${encodeURIComponent(dir)}`);
      setTree((current) => ({ ...current, [dir]: Array.isArray(data.items) ? data.items : [] }));
    } catch (error) {
      notify?.(error.message || "目录读取失败");
    } finally {
      setLoadingDirs((current) => { const next = new Set(current); next.delete(dir); return next; });
    }
  };
  const toggleDir = (dir) => {
    setExpandedDirs((current) => {
      const next = new Set(current);
      if (next.has(dir) && dir !== ".") next.delete(dir); else next.add(dir);
      return next;
    });
    if (!tree[dir]) void loadDirectory(dir);
  };
  const openFile = async (filePath) => {
    if (!projectId || fileBusy) return;
    setFileBusy(true);
    try {
      const data = await api(`/api/file?project_id=${encodeURIComponent(projectId)}&path=${encodeURIComponent(filePath)}`);
      if (data.error) throw new Error(data.error);
      setActivePath(filePath);
      setActiveContent(data.content || "");
      setActiveHash(data.sha256 || "");
    } catch (error) {
      notify?.(error.message || "文件读取失败");
    } finally {
      setFileBusy(false);
    }
  };
  const saveFile = async () => {
    if (!projectId || !activePath || fileBusy) return;
    setFileBusy(true);
    setTask?.({ state: "running", text: "正在保存文件..." });
    try {
      const body = { project_id: projectId, path: activePath, content: activeContent };
      if (activeHash) body.expected_sha256 = activeHash;
      const data = await api("/api/save", { method: "POST", body: jsonBody(body) });
      setActiveHash(data.sha256 || "");
      setTask?.({ state: "done", text: "文件已保存" });
      notify?.("文件已保存");
    } catch (error) {
      setTask?.({ state: "error", text: error.message || "保存失败" });
      notify?.(error.status === 409 ? "文件已变化，请重新读取后再保存" : (error.message || "保存失败"));
    } finally {
      setFileBusy(false);
    }
  };
  const updatePendingRuntime = (sessionId, updater) => {
    if (!sessionId) return;
    updateSessionMessages(sessionId, (current) => {
      const next = [...current];
      const last = next.at(-1);
      if (last?.role !== "assistant" || !last.pending) return current;
      next[next.length - 1] = updater(last);
      return next;
    });
  };
  const pauseAgent = () => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const sessionId = runningSessionIdRef.current || activeSession?.id;
    if (!sessionId || pauseRequestedRef.current) return;
    pauseRequestedRef.current = true;
    socket.send(JSON.stringify({ type: "pause" }));
    updatePendingRuntime(sessionId, (last) => ({
      ...last,
      runtime: true,
      runtimePaused: true,
      runtimeStatus: "已请求暂停，等待当前步骤结束",
      runtimeEvents: [...(last.runtimeEvents || []), { type: "pause", content: "用户请求暂停", at: Date.now() }].slice(-80),
    }));
    setTask?.({ state: "running", text: "Code Agent 暂停中..." });
  };
  const toggleRuntimeOpen = (messageId, index) => {
    if (!activeSession?.id) return;
    updateSessionMessages(activeSession.id, (current) => current.map((message, messageIndex) => {
      const isTarget = messageId ? message.id === messageId : messageIndex === index;
      return isTarget ? { ...message, runtimeOpen: !message.runtimeOpen } : message;
    }));
  };
  const normalizeAgentFinalContent = (content, type) => {
    const raw = String(content || "").trim();
    if (raw === "Agent paused by user.") return "已暂停";
    if (!raw) return type === "error" ? "Code Agent 失败" : "无回复";
    return content;
  };
  const renderCodeMessage = (item, index) => {
    if (item.runtime && item.pending) {
      const events = item.runtimeEvents || [];
      const status = item.runtimeStatus || (item.runtimePaused ? "暂停中" : "正在思考中");
      return <article key={item.id || `${item.role}-${index}`} className={`sa-code-message assistant sa-runtime-card ${item.runtimePaused ? "paused" : ""}`}>
        <button type="button" className="sa-runtime-cube" onClick={(event) => { event.stopPropagation(); pauseAgent(); }} disabled={item.runtimePaused} title={item.runtimePaused ? "已请求暂停" : "暂停 Agent"} aria-label={item.runtimePaused ? "已请求暂停" : "暂停 Agent"}>
          <Box size={18} />
          <Pause size={10} className="sa-runtime-pause-mark" />
        </button>
        <div className="sa-runtime-body">
          <span>Code Agent</span>
          <button type="button" className="sa-runtime-summary" onClick={() => toggleRuntimeOpen(item.id, index)} aria-expanded={Boolean(item.runtimeOpen)}>
            <strong>{status}</strong>
            {item.runtimeOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </button>
          {item.runtimeOpen && <div className="sa-runtime-events">
            {events.length ? events.map((event, eventIndex) => <div key={`${event.type || "event"}-${event.at || eventIndex}-${eventIndex}`}>
              <small>{formatTime(event.at)}</small>
              <p>{event.content || event.type || "运行事件"}</p>
            </div>) : <p>暂无运行细节</p>}
          </div>}
        </div>
      </article>;
    }
    return <article key={`${item.role}-${index}`} className={`sa-code-message ${item.role} ${item.error ? "error" : ""}`}>
      <span>{item.role === "user" ? "你" : "Code Agent"}</span>
      <p>{item.content || (item.pending ? "正在思考中..." : "")}</p>
      {item.tools?.length > 0 && <small>工具：{item.tools.map((tool) => tool.tool).filter(Boolean).join("、")}</small>}
    </article>;
  };
  const sendMessage = () => {
    const text = messageDraft.trim();
    if (!text || !projectId || agentBusy) return;
    const sessionId = activeSession?.id || createSession(codeSessionTitle(text));
    const history = (activeSession?.messages || []).slice(-10).map((item) => `${item.role === "user" ? "用户" : "Code Agent"}: ${item.content}`).join("\n");
    const reasoningMap = { low: "低", medium: "中", high: "高", max: "最高" };
    const message = [
      `当前项目：${selectedProject?.display_name || projectId}`,
      activePath ? `当前打开文件：${activePath}` : "当前打开文件：未选择，必要时请自行浏览项目目录",
      `推理强度：${reasoningMap[settings.reasoning] || settings.reasoning}`,
      settings.agentPrompt ? `Code Agent 附加提示词：\n${settings.agentPrompt}` : "",
      history ? `历史对话：\n${history}` : "",
      `本轮请求：${text}`,
    ].filter(Boolean).join("\n\n");
    setMessageDraft("");
    const runMessageId = `run-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    runningSessionIdRef.current = sessionId;
    pauseRequestedRef.current = false;
    updateSessionMessages(sessionId, (current) => [...current, { role: "user", content: text }, {
      id: runMessageId,
      role: "assistant",
      content: "",
      pending: true,
      runtime: true,
      runtimeOpen: false,
      runtimePaused: false,
      runtimeStatus: "正在思考中",
      runtimeEvents: [{ type: "start", content: "正在思考中", at: Date.now() }],
    }]);
    setAgentBusy(true);
    setTask?.({ state: "running", text: "Code Agent 正在思考中..." });
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/chat?token=${encodeURIComponent(getAuthToken())}`);
    socketRef.current = socket;
    socket.onopen = () => socket.send(JSON.stringify({
      message,
      original_message: text,
      domain: "code",
      project_id: projectId,
      active_path: activePath,
      token: getAuthToken(),
      session_id: sessionId,
      model: settings.model,
      base_url: settings.baseUrl,
      api_key: settings.apiKey,
      reasoning_effort: settings.reasoning,
      code_agent_prompt: settings.agentPrompt,
    }));
    socket.onmessage = (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.type === "status") setTask?.({ state: "running", text: data.content || "Code Agent 正在思考中..." });
      if (data.type === "status" || data.type === "paused") {
        const statusContent = data.content || (data.type === "paused" ? "已请求暂停，等待当前步骤结束" : "正在思考中");
        setTask?.({ state: "running", text: data.type === "paused" ? "Code Agent 暂停中..." : "Code Agent 正在思考中..." });
        updatePendingRuntime(sessionId, (last) => ({
          ...last,
          runtime: true,
          runtimePaused: data.type === "paused" || last.runtimePaused,
          runtimeStatus: statusContent,
          runtimeEvents: [...(last.runtimeEvents || []), { type: data.type, content: statusContent, at: Date.now() }].slice(-80),
        }));
        if (data.type === "paused") return;
      }
      if (data.type === "delta") return;
      if (data.type === "result" || data.type === "error") {
        const finalContent = normalizeAgentFinalContent(data.content, data.type);
        const wasPaused = finalContent === "已暂停";
        updateSessionMessages(sessionId, (current) => {
          const next = [...current];
          const last = next.at(-1);
          if (last?.role === "assistant") next[next.length - 1] = { ...last, content: finalContent || last.content || "无回复", pending: false, runtime: false, error: data.type === "error" && !wasPaused, tools: data.tool_calls || [] };
          return next;
        });
        setAgentBusy(false);
        runningSessionIdRef.current = "";
        pauseRequestedRef.current = false;
        setTask?.({ state: data.type === "error" && !wasPaused ? "error" : "done", text: wasPaused ? "Code Agent 已暂停" : (data.type === "error" ? finalContent : "Code Agent 完成") });
        void loadDirectory(".");
        if (activePath && data.type === "result") void openFile(activePath);
      }
    };
    socket.onerror = () => {
      setAgentBusy(false);
      runningSessionIdRef.current = "";
      pauseRequestedRef.current = false;
      updateSessionMessages(sessionId, (current) => {
        const next = [...current];
        const last = next.at(-1);
        if (last?.role === "assistant") next[next.length - 1] = { ...last, content: "Code Agent 连接失败", pending: false, error: true };
        return next;
      });
      setTask?.({ state: "error", text: "Code Agent 连接失败" });
    };
  };

  useEffect(() => {
    if (!projectId && normalizedProjects[0]) setProjectId(normalizedProjects[0].project_id);
  }, [normalizedProjects, projectId]);
  useEffect(() => {
    if (!projectId) return;
    window.localStorage.setItem("smart-harness-code-project-id", projectId);
    setTree({});
    setExpandedDirs(new Set(["."]));
    setActivePath("");
    setActiveContent("");
    setActiveHash("");
    void loadDirectory(".");
  }, [projectId]);
  useEffect(() => {
    if (!projectId) return;
    const latest = projectSessions[0];
    if (!latest) createSession();
    else if (!projectSessions.some((item) => item.id === activeSessionId)) {
      setActiveSessionId(latest.id);
      window.localStorage.setItem("smart-harness-code-session-id", latest.id);
    }
  }, [projectId, projectSessions.length, activeSessionId]);
  useEffect(() => () => { if (socketRef.current && socketRef.current.readyState < 2) socketRef.current.close(); }, []);
  const saveSettings = (next) => {
    // Browser credential managers sometimes autofill an account email into
    // the first text field of a settings dialog. Never accept that as a URL.
    const sanitized = { ...next };
    const baseUrl = String(sanitized.baseUrl || "").trim();
    sanitized.baseUrl = baseUrl && /^https?:\/\//i.test(baseUrl) ? baseUrl : "";
    setSettings(sanitized);
    const { apiKey: _apiKey, ...persisted } = sanitized;
    writeJsonStorage(CODE_AGENT_SETTINGS_KEY, persisted);
    notify?.("Code Agent 设置已更新");
  };
  const renderTree = (dir = ".", depth = 0) => {
    const rows = tree[dir] || [];
    return rows
      .slice()
      .sort((a, b) => (a.type === b.type ? a.name.localeCompare(b.name) : a.type === "dir" ? -1 : 1))
      .map((item) => {
        const fullPath = joinProjectPath(dir, item.name);
        const isDir = item.type === "dir";
        const isExpanded = expandedDirs.has(fullPath);
        return <div key={fullPath}>
          <button type="button" className={`sa-code-file-row ${activePath === fullPath ? "active" : ""}`} style={{ paddingLeft: 10 + depth * 14 }} onClick={() => isDir ? toggleDir(fullPath) : openFile(fullPath)}>
            {isDir ? (isExpanded ? <FolderOpen size={15} /> : <Folder size={15} />) : <FileText size={15} />}
            <span>{item.name}</span>
            {!isDir && <small>{Math.max(1, Math.ceil(Number(item.size || 0) / 1024))} KB</small>}
          </button>
          {isDir && isExpanded && <div>{loadingDirs.has(fullPath) ? <div className="sa-code-tree-loading">读取中...</div> : renderTree(fullPath, depth + 1)}</div>}
        </div>;
      });
  };

  return <section className="sa-code-shell">
    <div className="sa-code-project-bar">
      <label>当前项目<select value={projectId} onChange={(event) => setProjectId(event.target.value)} disabled={projectsLoading || !normalizedProjects.length}><option value="">{projectsLoading ? "读取项目中..." : "选择一个已导入项目"}</option>{normalizedProjects.map((item) => <option key={item.project_id} value={item.project_id}>{item.display_name || item.project_id}</option>)}</select></label>
      <span>{selectedProject ? `${selectedProject.status || "registered"} · ${selectedProject.artifact_count || 0} 个文件 · 修订 ${selectedProject.revision || 1}` : "Code Agent 每次只绑定一个项目"}</span>
    </div>
    {!projectId ? <EmptyPanel icon={<Folder size={28} />} title="选择项目" text="选中项目后，左侧目录、中间代码区和右侧对话都会绑定该项目。" /> : <div className="sa-code-ide">
      <aside className="sa-card sa-code-explorer"><div className="sa-code-pane-head"><div><h2>资源管理器</h2><p>{selectedProject?.display_name || projectId}</p></div><button type="button" className="sa-icon-button" onClick={() => loadDirectory(".")} title="刷新目录" aria-label="刷新目录"><RefreshCw size={15} className={loadingDirs.has(".") ? "spin" : ""} /></button></div><div className="sa-code-tree">{loadingDirs.has(".") && !tree["."] ? <EmptyState text="正在读取目录..." /> : renderTree(".")}{!loadingDirs.has(".") && tree["."]?.length === 0 && <EmptyState text="项目目录为空" />}</div></aside>
      <main className="sa-card sa-code-viewer"><div className="sa-code-pane-head"><div><h2>{activePath || "欢迎"}</h2><p>{activePath ? "当前文件只是工作区中的一个打开对象，Agent 可浏览整个项目" : "从左侧选择文件，或直接在右侧让 Code Agent 检查项目"}</p></div><button type="button" className="sa-button primary" disabled={!activePath || fileBusy} onClick={saveFile}>{fileBusy && <LoaderCircle size={14} className="spin" />}保存</button></div><textarea value={activeContent} onChange={(event) => setActiveContent(event.target.value)} disabled={!activePath || fileBusy} spellCheck="false" placeholder="选择文件后在这里查看和编辑代码" /></main>
      <aside className="sa-card sa-code-chat-panel"><div className="sa-code-chat-top"><div><h2>聊天</h2><p>{activeSession?.title || "新会话"}</p></div><div className="sa-actions compact"><button type="button" className="sa-icon-button" onClick={() => createSession()} title="新建会话" aria-label="新建会话"><Plus size={15} /></button><button type="button" className="sa-icon-button" onClick={() => setSettingsOpen(true)} title="Code Agent 设置" aria-label="Code Agent 设置"><Settings2 size={15} /></button></div></div>
        <div className="sa-code-side-grid"><div className="sa-code-session-list"><strong>历史对话</strong>{projectSessions.map((item) => <button type="button" key={item.id} className={activeSession?.id === item.id ? "active" : ""} onClick={() => { setActiveSessionId(item.id); window.localStorage.setItem("smart-harness-code-session-id", item.id); }}><span>{item.title}</span><small>{formatTime(item.updated_at)}</small></button>)}{!projectSessions.length && <span>暂无会话</span>}</div>
        <div className="sa-code-conversation"><div className="sa-code-messages">{activeMessages.map(renderCodeMessage)}{!activeMessages.length && <EmptyState text="像 Codex 一样描述要完成的代码任务" />}</div><div className="sa-code-composer"><textarea rows={3} value={messageDraft} onChange={(event) => setMessageDraft(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); sendMessage(); } }} disabled={agentBusy} placeholder="例如：扫描这个项目，找出启动失败原因并修复" /><button type="button" className="sa-button primary" onClick={sendMessage} disabled={!messageDraft.trim() || agentBusy}><Send size={15} />发送</button></div></div></div></aside>
    </div>}
    {settingsOpen && <CodeAgentSettingsDialog settings={settings} onSave={saveSettings} onClose={() => setSettingsOpen(false)} />}
  </section>;
}

function CodeAgentSettingsDialog({ settings, onSave, onClose }) {
  const [draft, setDraft] = useState(settings);
  const update = (field, value) => setDraft((current) => ({ ...current, [field]: value }));
  return <div className="sa-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="sa-modal sa-code-settings-modal" role="dialog" aria-modal="true" aria-labelledby="code-agent-settings-title">
      <header className="sa-modal-head"><div><h2 id="code-agent-settings-title">Code Agent 设置</h2><p>这些配置只影响 Code Agent 对当前项目的对话。</p></div><button type="button" className="sa-icon-button" onClick={onClose} title="关闭" aria-label="关闭"><X size={16} /></button></header>
      <div className="sa-code-settings-body" autoComplete="off"><label>模型<input name="code_agent_model" autoComplete="off" value={draft.model || ""} onChange={(event) => update("model", event.target.value)} placeholder="留空使用默认模型" /></label><label>Base URL<input name="code_agent_base_url" type="url" inputMode="url" autoComplete="off" value={draft.baseUrl || ""} onChange={(event) => update("baseUrl", event.target.value)} placeholder="https://..." /></label><label>API Key<input name="code_agent_api_key" type="password" autoComplete="new-password" value={draft.apiKey || ""} onChange={(event) => update("apiKey", event.target.value)} placeholder="仅本页临时使用，不写入 localStorage" /></label><label>推理强度<select name="code_agent_reasoning" autoComplete="off" value={draft.reasoning || "medium"} onChange={(event) => update("reasoning", event.target.value)}><option value="low">低：更快</option><option value="medium">中：默认</option><option value="high">高：更稳</option><option value="max">最高：复杂修改</option></select></label><label className="sa-code-settings-prompt">Code Agent 提示词<textarea name="code_agent_prompt" autoComplete="off" rows={8} value={draft.agentPrompt || ""} onChange={(event) => update("agentPrompt", event.target.value)} placeholder="例如：修改前先说明计划；只改当前项目；修改后必须运行测试。" /></label></div>
      <div className="sa-modal-actions"><button type="button" className="sa-button ghost" onClick={onClose}>取消</button><button type="button" className="sa-button primary" onClick={() => { onSave(draft); onClose(); }}>保存设置</button></div>
    </section>
  </div>;
}

function HistoryPanel({ entries, onSelect, selectedIds, onToggle, onToggleAll, allSelected, someSelected, onDelete, onDeleteSelected, deleting }) {
  const selectAllRef = useRef(null);
  useEffect(() => { if (selectAllRef.current) selectAllRef.current.indeterminate = someSelected && !allSelected; }, [someSelected, allSelected]);
  return <section className="sa-card"><div className="sa-card-head"><div><h2>岗位 JD 历史</h2><p>选择岗位 JD 后，会切换当前求职任务与该岗位关联的面试记录。</p></div><div className="sa-card-head-actions"><label className="sa-select-all"><input ref={selectAllRef} className="sa-checkbox" type="checkbox" checked={allSelected} onChange={onToggleAll} disabled={!entries.length || deleting} aria-label="全选岗位会话" />全选</label><span className="sa-badge">{entries.length} 条</span><button type="button" className="sa-button ghost" disabled={!selectedIds.size || deleting} onClick={onDeleteSelected}><Trash2 size={15} />删除已勾选（{selectedIds.size}）</button></div></div><div className="sa-history-list">{entries.map((entry) => <div className="sa-history-row" key={entry.id}><input className="sa-checkbox" type="checkbox" checked={selectedIds.has(entry.id)} onChange={() => onToggle(entry.id)} disabled={deleting} aria-label={`选择岗位会话 ${entry.title}`} /><button type="button" className="sa-history-entry" onClick={() => onSelect(entry)}><span className="sa-history-icon"><FileText size={16} /></span><div><strong>{entry.title}</strong><small>岗位 JD · {formatTime(entry.time)}</small></div><ChevronRight size={16} /></button><button type="button" className="sa-icon-button" title="删除岗位会话" aria-label={`删除岗位会话 ${entry.title}`} onClick={() => onDelete(entry)} disabled={deleting}><Trash2 size={15} /></button></div>)}{!entries.length && <EmptyState text="暂无岗位 JD 历史。" />}</div></section>;
}
function EmptyPanel({ icon, title, text }) { return <section className="sa-card sa-empty-panel"><div className="sa-empty-icon">{icon}</div><h2>{title}</h2><p>{text}</p></section>; }
function EmptyState({ text }) { return <div className="sa-empty-state">{text}</div>; }

export default function RootApp() { return <ErrorBoundary><App /></ErrorBoundary>; }
function ErrorBoundary({ children }) { const [error, setError] = useState(null); useEffect(() => { const handler = (event) => setError(event.error || new Error("页面出现异常")); window.addEventListener("error", handler); return () => window.removeEventListener("error", handler); }, []); if (error) return <div className="sa-fatal"><Settings2 size={28} /><h2>Career 页面暂时无法渲染</h2><p>{error.message}</p><button onClick={() => window.location.reload()} className="sa-button primary"><RefreshCw size={15} />重新加载</button></div>; return children; }
