import { ReportPage } from "@/features/xai/report-page";

export const metadata = {
  title: "Report",
};

export default async function ReportRoute({
  params,
}: {
  params: Promise<{ predictionId: string }>;
}) {
  const { predictionId } = await params;
  return <ReportPage predictionId={predictionId} />;
}
