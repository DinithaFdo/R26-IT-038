"use client";

import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SunMoon, Sun, Moon, Monitor } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

export function AppearanceSettings() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><SunMoon className="h-4 w-4" /> Appearance</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <div className="h-9 w-24 rounded-md bg-muted animate-pulse" />
            <div className="h-9 w-24 rounded-md bg-muted animate-pulse" />
            <div className="h-9 w-24 rounded-md bg-muted animate-pulse" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader><CardTitle className="flex items-center gap-2"><SunMoon className="h-4 w-4" /> Appearance</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Customize the console theme. System preference will follow your OS settings automatically.
        </p>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            className={cn(
              "w-28 gap-2 border-border/80",
              theme === "light" && "border-brand bg-brand/5 ring-1 ring-brand font-medium"
            )}
            onClick={() => setTheme("light")}
          >
            <Sun className="h-4 w-4" />
            Light
          </Button>
          <Button
            variant="outline"
            className={cn(
              "w-28 gap-2 border-border/80",
              theme === "dark" && "border-brand bg-brand/5 ring-1 ring-brand font-medium"
            )}
            onClick={() => setTheme("dark")}
          >
            <Moon className="h-4 w-4" />
            Dark
          </Button>
          <Button
            variant="outline"
            className={cn(
              "w-28 gap-2 border-border/80",
              theme === "system" && "border-brand bg-brand/5 ring-1 ring-brand font-medium"
            )}
            onClick={() => setTheme("system")}
          >
            <Monitor className="h-4 w-4" />
            System
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
