CUDA_ID=0,1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_ID} \
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --workers 16 \
  --batch-size 128 \
  --save-dir ~/ddsm_CLD/trial_1/c_50/eval \
  --pretrained ~/ddsm_CLD/trial_1/c_50/current.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --data ~/DDSM_patches