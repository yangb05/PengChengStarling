# export pretrained.pt for finetune
./zipformer/export.py \
  --epoch 75 \
  --avg 11 \
  --config-file config_train/example.yaml