import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";

// Subscribe to backend SSE; on any event, invalidate the jobs/gallery queries
// so React Query re-fetches and the UI re-renders. We deliberately don't
// reconcile event-by-event — the full GET is cheap and avoids drift bugs.
export function useJobsStream(): void {
  const qc = useQueryClient();
  useEffect(() => {
    const source = new EventSource("/api/events");
    const onAny = () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["gallery"] });
    };
    source.addEventListener("job_status", onAny);
    source.addEventListener("job_completed", onAny);
    source.addEventListener("job_failed", onAny);
    source.onerror = () => {
      // EventSource auto-reconnects; nothing to do.
    };
    return () => source.close();
  }, [qc]);
}
