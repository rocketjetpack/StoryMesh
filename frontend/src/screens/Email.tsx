import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../api/client";
import type { DraftSubmission } from "../App";

interface Props {
  value: string;
  draft: DraftSubmission;
  onBack: () => void;
  onSubmitted: (runId: string) => void;
}

const EMAIL_RE = /^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$/;

export default function EmailScreen({ value, draft, onBack, onSubmitted }: Props): JSX.Element {
  const [email, setEmail] = useState(value);
  const ref = useRef<HTMLInputElement | null>(null);
  useEffect(() => ref.current?.focus(), []);

  const valid = EMAIL_RE.test(email.trim());

  const submit = useMutation({
    mutationFn: () =>
      api.submit({
        prompt: draft.prompt,
        email: email.trim(),
        prompt_style: draft.promptStyle,
      }),
    onSuccess: (data) => onSubmitted(data.run_id),
  });

  return (
    <div>
      <p className="mono text-brass text-xs tracking-[0.3em] uppercase mb-4">Step 3 of 3 — Where to send it</p>
      <h2 className="display text-4xl md:text-5xl mb-3">Your email</h2>
      <p className="text-cream/60 mb-6">
        We email the finished PDF and EPUB the moment they're ready. We don't
        keep your address — and nobody at this booth will see it on screen.
      </p>

      <input
        ref={ref}
        type="email"
        autoComplete="email"
        spellCheck={false}
        className="prompt-field text-2xl"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="you@somewhere.com"
        onKeyDown={(e) => {
          if (e.key === "Enter" && valid && !submit.isPending) submit.mutate();
        }}
      />
      {submit.isError && (
        <p className="text-oxblood mt-3 text-sm font-mono">Submission failed: {String(submit.error)}</p>
      )}

      <div className="flex items-center justify-between mt-8">
        <button className="btn-ghost" onClick={onBack} disabled={submit.isPending}>
          ← Back
        </button>
        <button
          className="btn-primary"
          disabled={!valid || submit.isPending}
          onClick={() => submit.mutate()}
        >
          {submit.isPending ? "Sending to the press…" : "Send my book"}
        </button>
      </div>
    </div>
  );
}
