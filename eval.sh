export CUDA_VISIBLE_DEVICES="1"

zipformer/streaming_decode.py \
  --epoch $epoch \
  --avg $avg \
  --config-file config_train/example.yaml