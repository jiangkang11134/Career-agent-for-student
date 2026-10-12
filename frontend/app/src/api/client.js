const baseQueryKeys = {
  auth: ["auth", "me"],
  config: ["config"],
  agentModelConfig: ["career", "agent-model-config"],
  adminFlow: ["admin", "agent-flow"],
  workspace: ["career", "workspace"],
  artifacts: ["career", "artifacts"],
  facts: ["career", "facts", "candidate"],
  interviews: ["career", "interviews"],
  library: ["career", "material-library"],
  templates: ["career", "resume-templates"],
  projects: ["career", "project-library"],
  project: (id) => ["career", "project", id],
  material: (id) => ["career", "material", id],
};
export const queryKeys = new Proxy(baseQueryKeys, {
  get(target, name) {
    const value = target[name];
    const owner = getAuthSession()?.user?.user_id || "anonymous";
    return typeof value === "function" ? (...args) => [owner, ...value(...args)] : value ? [owner, ...value] : value;
  },
});

const AUTH_STORAGE_KEY = "career-auth-session";
const AUTH_EVENT = "career-auth-session-change";

export function getAuthSession() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(AUTH_STORAGE_KEY) || "{}");
    return parsed && typeof parsed.token === "string" && parsed.token ? parsed : null;
  } catch {
    return null;
  }
}

export function getAuthToken() {
  return getAuthSession()?.token || "";
}

export function setAuthSession(session) {
  if (!session?.token) return clearAuthSession();
  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify({
    token: session.token,
    user: session.user || null,
  }));
  window.dispatchEvent(new Event(AUTH_EVENT));
  return session;
}

export function clearAuthSession() {
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
  window.dispatchEvent(new Event(AUTH_EVENT));
  return null;
}

export function onAuthSessionChange(listener) {
  const handler = () => listener(getAuthSession());
  window.addEventListener(AUTH_EVENT, handler);
  window.addEventListener("storage", handler);
  return () => {
    window.removeEventListener(AUTH_EVENT, handler);
    window.removeEventListener("storage", handler);
  };
}

function apiErrorMessage(payload, status) {
  const detail = payload?.detail ?? payload?.error;
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (detail && typeof detail === "object") {
    if (typeof detail.message === "string" && detail.message.trim()) return detail.message.trim();
    try {
      return JSON.stringify(detail);
    } catch {
      return String(detail);
    }
  }
  return `请求失败 (${status})`;
}

export async function api(path, options = {}) {
  const { auth = true, ...fetchOptions } = options;
  const headers = new Headers(options.headers || {});
  const body = options.body;
  if (body != null && !(body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const token = auth ? getAuthToken() : "";
  if (token && !headers.has("Authorization")) headers.set("Authorization", "Bearer " + token);
  const method = String(fetchOptions.method || "GET").toUpperCase();
  if (method !== "GET" && method !== "HEAD" && method !== "OPTIONS" && !headers.has("Idempotency-Key")) {
    headers.set("Idempotency-Key", options.idempotencyKey || globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`);
  }
  const response = await fetch(path, { signal: AbortSignal.timeout(180000), ...fetchOptions, headers });
  if (auth && token !== getAuthToken()) throw new Error("登录状态已变化，请重新操作");
  const contentType = response.headers.get("content-type") || "";
  let payload;
  if (contentType.includes("application/json")) {
    payload = await response.json().catch(() => ({}));
  } else {
    const text = await response.text();
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    if (response.status === 401 && auth) clearAuthSession();
    const error = new Error(apiErrorMessage(payload, response.status));
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

export function jsonBody(value) {
  return JSON.stringify(value);
}

export function uploadBody(entries) {
  const body = new FormData();
  Object.entries(entries).forEach(([key, value]) => {
    if (Array.isArray(value)) value.forEach((item) => body.append(key, item));
    else if (value != null) body.append(key, value);
  });
  return body;
}
