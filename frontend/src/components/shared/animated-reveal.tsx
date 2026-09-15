"use client";

import { useEffect, useRef } from "react";

export function AnimatedReveal({ children, className }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion || !ref.current) return;
    let context: { revert: () => void } | null = null;
    void import("gsap").then(({ gsap }) => {
      if (!ref.current) return;
      context = gsap.context(() => {
        gsap.fromTo(
          ref.current,
          { autoAlpha: 0, y: 18 },
          { autoAlpha: 1, y: 0, duration: 0.55, ease: "power2.out" },
        );
      }, ref);
    });
    return () => context?.revert();
  }, []);

  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}
