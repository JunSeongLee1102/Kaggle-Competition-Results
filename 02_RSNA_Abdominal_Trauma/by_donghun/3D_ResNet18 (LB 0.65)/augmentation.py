# tio: 3D augmentation
augment = tio.Compose([
    tio.RandomAnisotropy(p=0.25),
    tio.RandomAffine(),
    tio.RandomFlip(),
    tio.RandomNoise(p=0.25),
    tio.RandomGamma(p=0.5),
])
