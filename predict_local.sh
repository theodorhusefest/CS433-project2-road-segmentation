export JOB_NAME="TEST1"
export GOOGLE_APPLICATION_CREDENTIALS="secret/CS433-59bd8aae4120.json"
export REGION="europe-west1"
export BUCKET_NAME="cs433-ml"
export JOB_DIR="gs://$BUCKET_NAME/keras-job-dir"
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

gcloud ai-platform local train   --package-path src   --module-name src.predict 