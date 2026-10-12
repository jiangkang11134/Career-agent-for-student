import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api, getAuthSession, jsonBody, onAuthSessionChange } from "../api/client";

const SessionContext = createContext(null);
const account = () => getAuthSession()?.user?.user_id || "";
const storageKey = (owner) => "career-workspace-session:" + owner;

export function SessionProvider({ children }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [owner, setOwner] = useState(account);
  const current = useRef(null);
  const generation = useRef(0);
  const queue = useRef(Promise.resolve());
  const accept = useCallback((value) => { current.current = value; setState(value); }, []);
  useEffect(() => onAuthSessionChange(() => setOwner(account())), []);
  useEffect(() => {
    const epoch = ++generation.current;
    accept(null);
    setError("");
    if (!owner) return;
    setPending(true);
    const initialize = async () => {
      const identity = sessionStorage.getItem(storageKey(owner));
      let value;
      try {
        value = identity ? await api("/api/career/contexts/" + identity) : null;
      } catch (failure) {
        if (failure.status !== 404) throw failure;
      }
      if (!value) value = await api("/api/career/contexts", { method: "POST", body: jsonBody({ feature: "career" }) });
      if (epoch !== generation.current) return;
      sessionStorage.setItem(storageKey(owner), value.session_id);
      accept(value);
    };
    queue.current = initialize().catch((failure) => {
      if (epoch === generation.current) setError(failure.message);
    }).finally(() => { if (epoch === generation.current) setPending(false); });
    return () => { generation.current++; };
  }, [owner, accept]);

  const update = useCallback((patch) => {
    const epoch = generation.current;
    queue.current = queue.current.then(async () => {
      if (epoch !== generation.current || !current.current) return;
      setPending(true);
      setError("");
      const before = current.current;
      try {
        const value = await api("/api/career/contexts/" + before.session_id, {
          method: "PATCH", body: jsonBody({ patch, expected_revision: before.context_revision, request_id: crypto.randomUUID() }),
        });
        if (epoch === generation.current) accept(value);
      } catch (failure) {
        if (epoch !== generation.current) return;
        setError(failure.message);
        if (failure.status === 409) {
          const restored = await api("/api/career/contexts/" + before.session_id);
          if (epoch === generation.current) accept(restored);
        }
      } finally { if (epoch === generation.current) setPending(false); }
    }).catch((failure) => { if (epoch === generation.current) setError(failure.message); });
    return queue.current;
  }, [accept]);
  const setStage = useCallback((stage) => update({ stage }), [update]);
  const selectRun = useCallback((id, stage) => update({ active_run_id: id || "", ...(stage ? { stage } : {}) }), [update]);
  const selectInterview = useCallback((interview, stage = "interview") => update({ active_interview_id: interview?.interview_id || "", stage }), [update]);
  const selectBundle = useCallback((id) => update({ input_bundle_id: id, stage: "interview" }), [update]);
  const newSession = useCallback(() => update({ active_run_id: "", stage: "evidence" }), [update]);
  const refresh = useCallback(async () => {
    const epoch = generation.current;
    if (!current.current) return;
    const value = await api("/api/career/contexts/" + current.current.session_id);
    if (epoch === generation.current) accept(value);
  }, [accept]);
  const value = useMemo(() => ({
    stage: state?.stage || "library", activeRunId: state?.active_run_id || "",
    activeInterviewId: state?.active_interview_id || "", inputBundleId: state?.input_bundle_id || "",
    workspaceSessionId: state?.session_id || "", contextRevision: state?.context_revision,
    coachMessages: state?.coach_messages || [],
    isNewSession: !state?.active_run_id, sessionEpoch: state?.session_id || "", pending, error,
    setStage, selectRun, selectInterview, selectBundle, newSession, refresh,
  }), [state, pending, error, setStage, selectRun, selectInterview, selectBundle, newSession, refresh]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSession must be used inside SessionProvider");
  return value;
}
