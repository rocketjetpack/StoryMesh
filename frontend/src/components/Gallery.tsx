import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { api, GalleryItem } from "../api/client";
import CoverDetailModal from "./CoverDetailModal";

export default function Gallery(): JSX.Element {
  const [selected, setSelected] = useState<GalleryItem | null>(null);

  const { data: items } = useQuery({
    queryKey: ["gallery"],
    queryFn: api.gallery,
    refetchInterval: 8000,
  });

  if (!items || items.length === 0) {
    return (
      <p className="text-cream/40 text-sm italic h-full flex items-center">
        Books printed today will appear here as they come off the press.
      </p>
    );
  }

  return (
    <>
      <div className="flex h-full gap-5 overflow-x-auto pb-1">
        <AnimatePresence initial={false}>
          {items.map((item) => (
            <CoverTile key={item.run_id} item={item} onClick={() => setSelected(item)} />
          ))}
        </AnimatePresence>
      </div>
      <CoverDetailModal item={selected} onClose={() => setSelected(null)} />
    </>
  );
}

function CoverTile({ item, onClick }: { item: GalleryItem; onClick: () => void }): JSX.Element {
  return (
    <motion.figure
      className="cover-tile cursor-pointer focus:outline-none focus:ring-2 focus:ring-brass/70 flex-shrink-0 h-full flex flex-col"
      style={{ width: "auto" }}
      layout
      initial={{ opacity: 0, rotateY: -90 }}
      animate={{ opacity: 1, rotateY: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onClick();
      }}
      aria-label={`Open synopsis for ${item.title}`}
    >
      <img
        src={item.cover_url}
        alt={item.title}
        loading="lazy"
        className="block h-full max-h-[78%] w-auto object-contain"
      />
      <figcaption className="display text-sm mt-2 leading-tight text-cream/90 px-1 max-w-[14rem] truncate">
        {item.title}
      </figcaption>
    </motion.figure>
  );
}
