export JOB_NAME="WED_1426_DROP05"
export GOOGLE_APPLICATION_CREDENTIALS="secret/CS433-59bd8aae4120.json"
export REGION="europe-west1"
export BUCKET_NAME="cs433-ml"
export JOB_DIR="gs://$BUCKET_NAME/keras-job-dir"

gcloud ai-platform jobs submit training $JOB_NAME   --package-path src/   --module-name src.run   --region $REGION   --python-version 3.5   --runtime-version 1.14   --job-dir $JOB_DIR   --stream-logs