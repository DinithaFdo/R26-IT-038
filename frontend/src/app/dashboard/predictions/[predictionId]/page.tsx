import { PredictionDetailPage } from "@/features/predictions/prediction-detail-page";

export default async function PredictionDetailRoute({
  params,
}: {
  params: Promise<{ predictionId: string }>;
}) {
  const { predictionId } = await params;
  return <PredictionDetailPage predictionId={predictionId} />;
}
