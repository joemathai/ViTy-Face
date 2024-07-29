import argparse
import torch
import deepspeed
import torchvision
import torch.distributed as dist
import torchvision.transforms as transforms
from loss.adaface import PartialFC_AdaFace
from models.vision_transformer import VisionTransformer


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

parser = argparse.ArgumentParser(description='DeepSpeed script')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--batch_size', type=int, default=12, help='')
parser.add_argument('--epochs', type=int, default=1000, help='')
parser.add_argument('--pretrained', type=bool, default=True, help='')
parser.add_argument('--dataset_root', type=str, default='/lfs_nvme/jmathai/webfaces/WebFace260M', help='')
# parser.add_argument('--dataset_root', type=str, default='/lfs_nvme/jmathai/webfaces/WebFace4M', help='')
parser.add_argument('--pretrained_weights', default='/lfs_nvme/jmathai/ViTy-Face/cache/dinov2_vitg14_reg4_pretrain.pth')
# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


deepspeed.init_distributed()

# make sure the seeds are fixed so each gpu sees the exact same batch
# this is needed for the AdaFace partial_fc implementation
torch.manual_seed(100)
torch.cuda.manual_seed(100)
# prepare dataset
transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=transforms.transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
             ])
train_dataset=torchvision.datasets.ImageFolder(root=args.dataset_root, transform=transforms)
train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True,
                     )

class VityFace(torch.nn.Module):
    def __init__(self, classnum, pretrained, pretrained_local_path):
        super().__init__()
        self.backbone = VisionTransformer(
                            patch_size=14,
                            embed_dim=1536,
                            depth=40,
                            num_heads=24,
                            mlp_ratio=4,
                            num_register_tokens=4,
                        )
        if pretrained:
            state_dict = torch.load(pretrained_local_path)
            print(f'loading backbone from pretrained weights rank:{dist.get_rank()} path:{pretrained_local_path}')
            self.backbone.load_state_dict(state_dict)
        self.adaface_loss = PartialFC_AdaFace(embedding_size=1536, classnum=classnum, 
                                              rank=dist.get_rank(), world_size=dist.get_world_size())

    def forward(self, batch):
        x, y = batch
        outputs = self.backbone(x)
        loss = self.adaface_loss(outputs['x_norm_clstoken'], y)
        return loss


model = VityFace(classnum=len(train_dataset.classes), pretrained=args.pretrained, pretrained_local_path=args.pretrained_weights)
model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters())

# TODO: write better training loop
# TODO: write eval pipeline to validate
# TODO: write better loggin of losses and norms
# TODO: write model checkpointing and fix the ds_config.json

for epoch in range(args.epochs):
    print('------ epoch: {epoch} ------')
    for step, batch in enumerate(train_dataloader):
        batch = (batch[0].to(model_engine.local_rank), batch[1].to(model_engine.local_rank))
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
