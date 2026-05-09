import { useEffect } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { api, GalleryItem } from "../api/client";

interface Props {
  item: GalleryItem | null;
  onClose: () => void;
}

export default function CoverDetailModal({ item, onClose }: Props): JSX.Element {
  // Esc closes; locks scroll while open.
  useEffect(() => {
    if (!item) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [item, onClose]);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["synopsis", item?.run_id],
    queryFn: () => api.runSynopsis(item!.run_id),
    enabled: !!item,
    staleTime: 60_000,
  });

  // Rendered into document.body via a portal so the modal escapes any
  // ancestor with `transform`, `filter`, or `backdrop-filter` (the side
  // panel uses `backdrop-blur-sm`, which would otherwise become the
  // containing block for `position: fixed` and trap the modal in the panel).
  return createPortal(
    <AnimatePresence>
      {item && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/70 backdrop-blur-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25 }}
          onClick={onClose}
        >
          <motion.div
            className="relative max-w-4xl w-full max-h-[88vh] overflow-y-auto bg-[var(--ink-soft)] border border-brass/40 rounded-sm grid grid-cols-1 md:grid-cols-[minmax(0,40%)_minmax(0,60%)] gap-0"
            initial={{ opacity: 0, y: 24, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.98 }}
            transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="absolute top-3 right-3 z-10 mono text-cream/60 hover:text-brass-bright text-xs tracking-[0.18em] uppercase px-2 py-1"
              onClick={onClose}
              aria-label="Close"
            >
              Close ×
            </button>

            <div className="bg-black/30 p-6 flex items-center justify-center">
              <img
                src={item.cover_url}
                alt={item.title}
                className="max-h-[60vh] w-auto shadow-[0_18px_60px_-12px_rgba(0,0,0,0.6)]"
              />
            </div>

            <div className="p-8 flex flex-col">
              <p className="mono text-brass text-[0.65rem] tracking-[0.3em] uppercase mb-3">
                Back cover
              </p>
              <h3 className="display text-3xl mb-5 leading-tight text-cream">{item.title}</h3>
              <div className="gold-rule mb-5" />
              {isLoading && <p className="text-cream/40 italic">Fetching the back cover…</p>}
              {isError && (
                <p className="text-oxblood font-mono text-sm">Synopsis unavailable for this run.</p>
              )}
              {data && (
                <p className="text-cream/85 leading-relaxed text-lg whitespace-pre-line">
                  {data.synopsis}
                </p>
              )}
              <div className="mt-auto pt-6 mono text-cream/30 text-[0.7rem]">
                run · {item.run_id.slice(0, 8)}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  );
}
