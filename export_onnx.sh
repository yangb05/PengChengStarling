# export onnx for deployment
python zipformer/export-onnx-streaming.py \
  --epoch 75 \
  --avg 11 \
  --config-file config_train/example.yaml