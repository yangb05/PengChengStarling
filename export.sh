# export pretrained.pt for finetune
./zipformer/export.py \
  --epoch 43 \
  --avg 15 \
  --config-file config_run/example.yaml