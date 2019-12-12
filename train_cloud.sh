export JOB_NAME="LAY3_SIZE256_BATCH16"
export GOOGLE_APPLICATION_CREDENTIALS="secret/CS433-59bd8aae4120.json"
export REGION="europe-west1"
export BUCKET_NAME="cs433-ml"
export JOB_DIR="gs://$BUCKET_NAME/keras-job-dir"
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

gcloud ai-platform jobs submit training $JOB_NAME --package-path src/   --module-name src.run --region $REGION  --python-version 3.5 --runtime-version 1.14  --job-dir $JOB_DIR  --stream-logs 