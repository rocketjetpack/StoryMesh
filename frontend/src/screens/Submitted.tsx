import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { api } from "../api/client";

interface Props {
  runId: string;
  resetAfterSeconds: number;
  onReset: () => void;
}

export default function Submitted({ runId, resetAfterSeconds, onReset }: Props): JSX.Element {
  const [secondsLeft, setSecondsLeft] = useState(resetAfterSeconds);

  // Poll our own run to surface a working title once it's known.
  const { data: jobs } = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: 2000,
  });
  const myJob = jobs?.find((j) => j.run_id === runId);

  useEffect(() => {
    if (secondsLeft <= 0) {
      onReset();
      return;
    }
    const t = window.setTimeout(() => setSecondsLeft((s) => s - 1), 1000);
    return () => window.clearTimeout(t);
  }, [secondsLeft, onReset]);

  return (
    <div className="text-center">
      <motion.p
        className="mono text-brass text-xs tracking-[0.3em] uppercase mb-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        On the press
      </motion.p>
      <motion.h2
        className="display text-5xl md:text-6xl mb-6"
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
      >
        Thank you.
      </motion.h2>
      <motion.div
        className="text-cream/80 text-xl max-w-xl mx-auto leading-relaxed"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.6 }}
      >
        {myJob?.title ? (
          <>
            We've started work on <span className="italic text-brass-bright">{myJob.title}</span>.
          </>
        ) : (
          <>We're sketching the outline now.</>
        )}
        <br />
        <span className="text-cream/60 text-base mt-3 inline-block">
          Watch the press in the side panel — your book will be emailed when it's bound.
        </span>
      </motion.div>

      <motion.div
        className="mono text-cream/40 text-xs mt-12"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.9 }}
      >
        ID {runId.slice(0, 8)} · resetting in {secondsLeft}s
      </motion.div>
      <button className="btn-ghost mt-2" onClick={onReset}>
        Make another →
      </button>
    </div>
  );
}
