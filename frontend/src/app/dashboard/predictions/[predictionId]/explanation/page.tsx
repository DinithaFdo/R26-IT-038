import { ExplanationPage } from "@/features/xai/explanation-page";

export const metadata = {
  title: "Explanation",
};

export default async function ExplanationRoute({
  params,
}: {
  params: Promise<{ predictionId: string }>;
}) {
  const { predictionId } = await params;
  return <ExplanationPage predictionId={predictionId} />;
}
