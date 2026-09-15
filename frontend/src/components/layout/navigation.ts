import {
  Activity,
  Braces,
  FileText,
  Gauge,
  History,
  Home,
  Microscope,
  Settings,
  Upload,
} from "lucide-react";

export const dashboardNavItems = [
  { href: "/dashboard", label: "Overview", icon: Home },
  { href: "/dashboard/analyze", label: "Analyze", icon: Upload },
  { href: "/dashboard/text-analyze", label: "Text Analysis", icon: FileText },
  { href: "/dashboard/history", label: "History", icon: History },
  { href: "/dashboard/system", label: "System", icon: Gauge },
  { href: "/dashboard/developer", label: "Developer", icon: Braces },
  { href: "/dashboard/settings", label: "Settings", icon: Settings },
];

export const landingFeatures = [
  {
    icon: Activity,
    title: "Multi-branch classification",
    description: "Branch outputs remain visible so researchers can inspect how each classifier contributed.",
  },
  {
    icon: Upload,
    title: "Audio format support",
    description: "Upload common voice formats or submit a completed browser recording for backend validation.",
  },
  {
    icon: Microscope,
    title: "Research transparency",
    description: "Dummy, mixed, and real model modes are clearly separated from research-eligible outputs.",
  },
];
