import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import { DashboardShell } from "@/components/layout/dashboard-shell";

export const metadata = {
  title: "Dashboard",
};

export default async function DashboardLayout({ children }: { children: React.ReactNode }) {
  const hasPublishableKey = Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);
  const hasSecretKey = Boolean(process.env.CLERK_SECRET_KEY);

  if (hasPublishableKey && hasSecretKey) {
    const { userId, redirectToSignIn } = await auth();
    if (!userId) {
      return redirectToSignIn({ returnBackUrl: "/dashboard" });
    }
  } else if (process.env.NODE_ENV === "production") {
    // Fail closed. A production build with incomplete Clerk configuration must
    // never render owner-scoped dashboard routes unauthenticated; previously a
    // missing publishable key skipped the check entirely.
    redirect("/sign-in");
  } else if (hasPublishableKey && !hasSecretKey) {
    redirect("/sign-in");
  }

  return <DashboardShell>{children}</DashboardShell>;
}
