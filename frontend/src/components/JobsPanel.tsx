import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { api, JobStatus } from "../api/client";

const STAGE_LABEL: Record<string, string> = {
  genre_normalizer: "Reading the genre",
  book_fetcher: "Consulting the library",
  book_ranker: "Ranking influences",
  theme_extractor: "Finding the themes",
  proposal_draft: "Drafting the proposal",
  rubric_judge: "Editorial review",
  proposal_reader_feedback: "Reader feedback",
  story_writer: "Writing the book",
  resonance_reviewer: "Polishing the prose",
  cover_art: "Painting the cover",
  book_assembler: "Binding the book",
};

export default function JobsPanel(): JSX.Element {
  const { data: jobs, isLoading } = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: 5000,
  });

  const visible = jobs ?? [];

  if (isLoading) return <p className="text-cream/40 text-sm">Loading…</p>;
  if (visible.length === 0) {
    return <p className="text-cream/40 text-sm italic">The press is idle. Be the first.</p>;
  }
  return (
    <ul className="space-y-3">
      <AnimatePresence initial={false}>
        {visible.map((job) => (
          <JobCard key={job.run_id} job={job} />
        ))}
      </AnimatePresence>
    </ul>
  );
}

function JobCard({ job }: { job: JobStatus }): JSX.Element {
  const stageLabel = job.stage ? STAGE_LABEL[job.stage] ?? job.stage : "Queued";
  const isComplete = job.status === "completed";
  const isFailed = job.status === "failed";
  const isTerminal = isComplete || isFailed;

  // Progress: completed runs render full bar; failed shows where it stopped.
  const rawProgress = job.total_stages > 0 ? job.stage_index / job.total_stages : 0;
  const progress = isComplete ? 1 : rawProgress;

  const borderColor = isComplete
    ? "border-patina/70"
    : isFailed
      ? "border-oxblood-bright/70"
      : "border-brass/60";

  const barColor = isComplete ? "bg-patina" : isFailed ? "bg-oxblood-bright" : "bg-brass";

  return (
    <motion.li
      layout
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
      className={`border-l-2 ${borderColor} pl-3 py-2 ${isTerminal ? "opacity-80" : ""}`}
    >
      <div className="flex items-center gap-2 mb-1">
        <StatusDot status={job.status} />
        <span className="display text-cream text-base truncate">
          {job.title ?? <span className="italic text-cream/50">Deciphering title…</span>}
        </span>
      </div>
      <div className="flex items-center justify-between text-xs">
        <span className="mono text-cream/60">{isTerminal ? terminalLabel(job.status) : stageLabel}</span>
        <StatusBadge job={job} />
      </div>
      <div className="h-[2px] bg-cream/10 mt-2 overflow-hidden">
        <motion.div
          className={`h-full ${barColor}`}
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(2, progress * 100)}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
      </div>
    </motion.li>
  );
}

function StatusDot({ status }: { status: JobStatus["status"] }): JSX.Element {
  if (status === "running") {
    return <span className="ink-dot animate-ink-pulse" aria-hidden />;
  }
  if (status === "completed") {
    return <span className="ink-dot bg-patina-bright" aria-hidden />;
  }
  if (status === "failed") {
    return <span className="ink-dot bg-oxblood-bright" aria-hidden />;
  }
  return <span className="ink-dot opacity-30" aria-hidden />;
}

function StatusBadge({ job }: { job: JobStatus }): JSX.Element | null {
  if (job.status === "completed") {
    return (
      <span className="mono text-[0.65rem] tracking-[0.18em] uppercase text-patina-bright">
        Complete
      </span>
    );
  }
  if (job.status === "failed") {
    return (
      <span className="mono text-[0.65rem] tracking-[0.18em] uppercase text-oxblood-bright">
        Error
      </span>
    );
  }
  if (job.status === "queued" && job.queue_position !== null) {
    return <span className="mono text-cream/40">queue #{job.queue_position}</span>;
  }
  if (job.status === "running") {
    return (
      <span className="mono text-cream/40">
        {job.stage_index}/{job.total_stages}
      </span>
    );
  }
  return null;
}

function terminalLabel(status: JobStatus["status"]): string {
  if (status === "completed") return "Pressed and bound";
  if (status === "failed") return "Did not finish";
  return "";
}
