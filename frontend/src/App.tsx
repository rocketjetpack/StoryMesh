import { useCallback, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useJobsStream } from "./hooks/useJobsStream";
import { useIdleReset } from "./hooks/useIdleReset";
import Welcome from "./screens/Welcome";
import Prompt from "./screens/Prompt";
import Style from "./screens/Style";
import EmailScreen from "./screens/Email";
import Submitted from "./screens/Submitted";
import JobsPanel from "./components/JobsPanel";
import Gallery from "./components/Gallery";

const IDLE_SECONDS = 90;
const POST_SUBMIT_RESET_SECONDS = 12;

type Screen = "welcome" | "prompt" | "style" | "email" | "submitted";

export interface DraftSubmission {
  prompt: string;
  promptStyle: string;
  email: string;
  runId?: string;
  title?: string;
}

const emptyDraft = (): DraftSubmission => ({ prompt: "", promptStyle: "default", email: "" });

export default function App(): JSX.Element {
  const [screen, setScreen] = useState<Screen>("welcome");
  const [draft, setDraft] = useState<DraftSubmission>(emptyDraft);

  useJobsStream();

  const reset = useCallback(() => {
    setDraft(emptyDraft());
    setScreen("welcome");
  }, []);

  useIdleReset(screen !== "welcome" && screen !== "submitted", IDLE_SECONDS, reset);

  return (
    <div className="h-screen w-full grid grid-rows-[minmax(0,62fr)_minmax(0,38fr)]">
      {/* Top zone: full-width wizard, atmospheric breathing room */}
      <main className="relative flex flex-col items-center justify-center p-10 min-h-0 overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={screen}
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            className="w-full max-w-3xl"
          >
            {screen === "welcome" && <Welcome onBegin={() => setScreen("prompt")} />}
            {screen === "prompt" && (
              <Prompt
                value={draft.prompt}
                onBack={reset}
                onContinue={(prompt) => {
                  setDraft((d) => ({ ...d, prompt }));
                  setScreen("style");
                }}
              />
            )}
            {screen === "style" && (
              <Style
                value={draft.promptStyle}
                onBack={() => setScreen("prompt")}
                onContinue={(promptStyle) => {
                  setDraft((d) => ({ ...d, promptStyle }));
                  setScreen("email");
                }}
              />
            )}
            {screen === "email" && (
              <EmailScreen
                value={draft.email}
                onBack={() => setScreen("style")}
                draft={draft}
                onSubmitted={(runId) => {
                  setDraft((d) => ({ ...d, runId }));
                  setScreen("submitted");
                }}
              />
            )}
            {screen === "submitted" && (
              <Submitted
                runId={draft.runId ?? ""}
                resetAfterSeconds={POST_SUBMIT_RESET_SECONDS}
                onReset={reset}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Bottom zone: Library (65%) + In the Press (35%) */}
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,65fr)_minmax(0,35fr)] border-t border-brass/20 bg-black/30 backdrop-blur-sm min-h-0">
        <section className="px-10 py-5 flex flex-col min-h-0">
          <div className="flex items-baseline gap-4 mb-3">
            <h2 className="display text-xl tracking-wide text-cream">Today's Library</h2>
            <p className="text-cream/60 text-sm">Books made at this booth — tap a cover for the synopsis</p>
          </div>
          <div className="flex-1 min-h-0">
            <Gallery />
          </div>
        </section>

        <aside className="border-l border-white/5 bg-black/20 px-8 py-5 flex flex-col min-h-0">
          <div className="mb-3">
            <h2 className="display text-xl tracking-wide text-cream">In the Press</h2>
            <p className="text-cream/60 text-sm mt-1">Live pipeline runs</p>
          </div>
          <div className="flex-1 overflow-y-auto pr-1 min-h-0">
            <JobsPanel />
          </div>
        </aside>
      </div>
    </div>
  );
}
