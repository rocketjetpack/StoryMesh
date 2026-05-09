import { motion } from "framer-motion";

interface Props {
  onBegin: () => void;
}

export default function Welcome({ onBegin }: Props): JSX.Element {
  return (
    <div className="text-center">
      <motion.p
        className="mono text-brass text-xs tracking-[0.3em] uppercase mb-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.6 }}
      >
        StoryMesh — A Live Press
      </motion.p>
      <motion.h1
        className="display text-7xl md:text-8xl mb-8 leading-[1.05]"
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35, duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
      >
        Tell us a story idea
        <br />
        <span className="text-brass-bright italic">we'll write it.</span>
      </motion.h1>
      <motion.p
        className="text-cream/70 text-xl max-w-xl mx-auto mb-12"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, duration: 0.6 }}
      >
        A short prompt becomes an original novel-shaped book. Cover art,
        layout, the whole thing — emailed to you when it's done.
      </motion.p>
      <motion.button
        className="btn-primary"
        onClick={onBegin}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.85, duration: 0.5 }}
        whileHover={{ scale: 1.03 }}
        whileTap={{ scale: 0.98 }}
      >
        Begin
      </motion.button>
    </div>
  );
}
