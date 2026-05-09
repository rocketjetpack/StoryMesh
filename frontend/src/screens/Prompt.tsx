import { useEffect, useRef, useState } from "react";

interface Props {
  value: string;
  onContinue: (prompt: string) => void;
  onBack: () => void;
}

const SAMPLE_PROMPTS = [
  "A retired conservator in Lisbon takes one last commission and finds a forgery she signed thirty years ago.",
  "Two siblings inherit a lighthouse on a coast that no longer has a sea.",
  "A bureaucrat at the world's smallest department of weights and measures discovers something is missing from the unit catalogue.",
];

export default function Prompt({ value, onContinue, onBack }: Props): JSX.Element {
  const [text, setText] = useState(value);
  const ref = useRef<HTMLTextAreaElement | null>(null);
  useEffect(() => ref.current?.focus(), []);

  const ready = text.trim().length >= 12;

  return (
    <div>
      <p className="mono text-brass text-xs tracking-[0.3em] uppercase mb-4">Step 1 of 3 — Your premise</p>
      <h2 className="display text-4xl md:text-5xl mb-3">What should the book be about?</h2>
      <p className="text-cream/60 mb-6">
        A sentence is plenty. A paragraph is welcome. Think setting, mood, a character or a moment.
      </p>

      <textarea
        ref={ref}
        className="prompt-field min-h-[180px]"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={SAMPLE_PROMPTS[Math.floor(Math.random() * SAMPLE_PROMPTS.length)]}
        maxLength={2000}
        rows={5}
      />
      <div className="flex items-center justify-between mt-8">
        <button className="btn-ghost" onClick={onBack}>
          ← Start over
        </button>
        <button
          className="btn-primary"
          disabled={!ready}
          onClick={() => onContinue(text.trim())}
        >
          Continue
        </button>
      </div>
    </div>
  );
}
