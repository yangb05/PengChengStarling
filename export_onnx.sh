# export onnx for deployment
python zipformer/export-onnx-streaming.py \
  --epoch 43 \
  --avg 15 \
  --config-file config_train/example.yaml