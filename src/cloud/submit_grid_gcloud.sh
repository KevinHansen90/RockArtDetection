#!/usr/bin/env bash
# Submit 20 pilot matrix jobs to GCP Vertex AI using gcloud CLI with active-job deduplication

PROJECT_ID=${1:-"${GCP_PROJECT_ID:-your-gcp-project-id}"}
REGION=${2:-"${GCP_REGION:-us-central1}"}
IMAGE_URI=${3:-"${ARTIFACT_REGISTRY_REPO:-us-central1-docker.pkg.dev/${PROJECT_ID}/rockart-docker-repo/rockart-trainer:latest}"}
GCS_BUCKET=${4:-"${GCS_BUCKET_NAME:-${PROJECT_ID}-rockart-data}"}
SERVICE_ACCOUNT=${5:-"${VERTEX_SERVICE_ACCOUNT:-rockart-vertex-sa@${PROJECT_ID}.iam.gserviceaccount.com}"}

MODELS=("fasterrcnn" "retinanet" "deformable_detr" "yolov5")
FILTERS=("base" "bilateral" "unsharp" "laplacian" "clahe")

echo "[*] Fetching currently running/pending jobs on Vertex AI for deduplication..."
EXISTING_JOBS=$(gcloud ai custom-jobs list --region="${REGION}" --filter="state=JOB_STATE_PENDING OR state=JOB_STATE_RUNNING" --format="value(displayName)" 2>/dev/null)

for MODEL in "${MODELS[@]}"; do
    for FILTER in "${FILTERS[@]}"; do
        JOB_NAME="pilot_${MODEL}_${FILTER}"

        if echo "${EXISTING_JOBS}" | grep -q "^${JOB_NAME}$"; then
            echo "[SKIP] ${JOB_NAME} is already active/running on Vertex AI."
            continue
        fi

        echo "Submitting ${JOB_NAME}..."

        MACHINE_TYPE="n1-standard-4"
        if [ "${MODEL}" == "deformable_detr" ]; then
            MACHINE_TYPE="n1-standard-8"
        fi

        gcloud ai custom-jobs create \
            --region="${REGION}" \
            --display-name="${JOB_NAME}" \
            --worker-pool-spec=container-image-uri="${IMAGE_URI}",machine-type=${MACHINE_TYPE},replica-count=1 \
            --args="model=${MODEL},data_root=gs://${GCS_BUCKET}/data/tiles/${FILTER},classes_file=gs://${GCS_BUCKET}/data/grouped_classes.txt,train=cpu_pilot,experiment=${JOB_NAME}" \
            --service-account="${SERVICE_ACCOUNT}"
    done
done

echo "[+] Deduplication check & submission complete."
