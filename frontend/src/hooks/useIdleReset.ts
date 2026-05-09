import { useEffect, useRef } from "react";

// When `enabled` is true, invoke `onIdle` after `seconds` of no user input.
// Any pointer/keyboard activity restarts the timer.
export function useIdleReset(enabled: boolean, seconds: number, onIdle: () => void): void {
  const timer = useRef<number | null>(null);
  const onIdleRef = useRef(onIdle);
  onIdleRef.current = onIdle;

  useEffect(() => {
    if (!enabled) return;
    const reset = () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => onIdleRef.current(), seconds * 1000);
    };
    reset();
    const events: (keyof WindowEventMap)[] = ["pointerdown", "keydown", "touchstart"];
    events.forEach((e) => window.addEventListener(e, reset, { passive: true }));
    return () => {
      events.forEach((e) => window.removeEventListener(e, reset));
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, [enabled, seconds]);
}
