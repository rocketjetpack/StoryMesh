import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { api, PromptStyleOption } from "../api/client";

interface Props {
  value: string;
  onContinue: (promptStyleId: string) => void;
  onBack: () => void;
}

export default function Style({ value, onContinue, onBack }: Props): JSX.Element {
  const { data: styles, isLoading } = useQuery({
    queryKey: ["prompt-styles"],
    queryFn: api.promptStyles,
    staleTime: 60_000,
  });
  const [selected, setSelected] = useState<string>(value);

  // First time the screen renders, default to the recommended style.
  useEffect(() => {
    if (selected !== "" && styles?.some((s) => s.id === selected)) return;
    if (styles && styles.length > 0) {
      const rec = styles.find((s) => s.is_recommended);
      setSelected(rec?.id ?? styles[0].id);
    }
  }, [styles, selected]);

  return (
    <div>
      <p className="mono text-brass text-xs tracking-[0.3em] uppercase mb-4">Step 2 of 3 — How we'll write it</p>
      <h2 className="display text-4xl md:text-5xl mb-3">Choose a writing approach.</h2>
      <p className="text-cream/60 mb-6">
        Each approach steers the model differently. Pick one — or keep the recommended.
      </p>

      {isLoading && <p className="text-cream/40">Loading approaches…</p>}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[55vh] overflow-y-auto pr-2">
        {styles?.map((s, i) => (
          <StyleCard
            key={s.id}
            style={s}
            selected={selected === s.id}
            onSelect={() => setSelected(s.id)}
            index={i}
          />
        ))}
      </div>

      <div className="flex items-center justify-between mt-8">
        <button className="btn-ghost" onClick={onBack}>
          ← Back
        </button>
        <button className="btn-primary" disabled={!selected} onClick={() => onContinue(selected)}>
          Continue
        </button>
      </div>
    </div>
  );
}

function StyleCard({
  style,
  selected,
  onSelect,
  index,
}: {
  style: PromptStyleOption;
  selected: boolean;
  onSelect: () => void;
  index: number;
}): JSX.Element {
  return (
    <motion.div
      className="voice-card"
      data-selected={selected}
      data-recommended={style.is_recommended}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onSelect();
      }}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.04 * index, duration: 0.4 }}
    >
      <div className="display text-lg mb-1">{style.name}</div>
      <div className="text-cream/70 text-sm leading-snug">{style.description}</div>
    </motion.div>
  );
}
