# 3D ResNet18 (LB 0.65)

* CV 0.5508, LB 0.65 (at epoch 50)
* self.avgpool = nn.AvgPool3d(kernel_size=(5, 5, 5), stride=(3, 3, 3), padding=(1, 1, 1))로 변경
* 3D Augmentation
https://github.com/JunSeongLee1102/Kaggle-Competition-Results/blob/61363ad0b993fd118b470a6f617084ef8ad06363/02_RSNA_Abdominal_Trauma/by_donghun/3D_ResNet18%20(LB%200.65)/augmentation.py#L2-L8
