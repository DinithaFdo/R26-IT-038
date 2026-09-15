"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import GradientWaves from "@/components/ui/GradientWaves";

export function HeroBackground() {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  const isDark = resolvedTheme === "dark";

  return (
    <div className="absolute inset-0 z-0 pointer-events-none" style={{ height: '100%' }}>
      <GradientWaves
        // International Orange palette
        horizonColor={isDark ? "#461304" : "#ffecd3"}  // 950 dark, 100 light
        waveColor={isDark ? "#cc3e02" : "#ff700a"}     // 700 dark, 500 light
        crestColor={isDark ? "#ff9032" : "#fff7ec"}    // 400 dark, 50 light
        speed={0.4}
        amplitude={2.5}
        waveScale={0.6}
        waveRatio={0.9}
        swell={35}
        turbulence={20}
        tilt={1.11}
        zoom={1.0}
        height={5.5}
        fogDepth={15}
        detail="medium"
        brightness={1.0}
        opacity={1.0}
        mouseInteraction={true}
        parallaxStrength={0.5}
        grain={true}
        grainIntensity={0.05}
      />
    </div>
  );
}
