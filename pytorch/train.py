import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from data.data import InpaintingDataset, ToTensor
from data.dataset_tfrecord import define_dataset
from model.net import InpaintingModel_GMCNN
from options.train_options import TrainOptions
from util.utils import getLatest

config = TrainOptions().parse()

print('loading data..')
# Original Dataset
# dataset = InpaintingDataset(config.dataset_path, '', transform=transforms.Compose([
#     ToTensor()
# ]))
# dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)

# Modified Dataset for Skin Inpainting
tfrecord_path = "gs://labelling-tools-data/tfrecords/person-tfrecord-v1.5_canva_improved_grapy.record"

cmd = f"gsutil -m cp -r {tfrecord_path} /content/"
if not os.path.exists(os.path.join('/content/', tfrecord_path.split('/')[-1])):
    os.system(cmd)

tfrecord_path = os.path.join('/content/', tfrecord_path.split('/')[-1])
trainset, trainset_length = define_dataset(tfrecord_path, config.batch_size, train=True)
# valset, valset_length = define_dataset(tfrecord_path, config.batch_size, train=False)
print('data loaded..')

print('configuring model..')
ourModel = InpaintingModel_GMCNN(in_channels=5, opt=config)
ourModel.print_networks()
if config.load_model_dir != '':
    print('Loading pretrained model from {}'.format(config.load_model_dir))
    ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
    print('Loading done.')
# ourModel = torch.nn.DataParallel(ourModel).cuda()
print('model setting up..')
print('training initializing..')
writer = SummaryWriter(log_dir=config.model_folder)
cnt = 0
for epoch in range(config.epochs):

    train_iterator = iter(trainset)
    num_iterations = int(trainset_length/config.batch_size)

    # for i, data in enumerate(dataloader):
    for i in range(num_iterations):
        # gt = data['gt'].cuda()
        # # normalize to values between -1 and 1
        # gt = gt / 127.5 - 1

        data, model_inputs = next(train_iterator)

        # TF to Torch data conversion
        inpaint_region = torch.tensor(data['inpaint_region'].numpy()).float().cuda().permute(0, 3, 1, 2)
        person = torch.tensor(data['person'].numpy()).float().cuda().permute(0, 3, 1, 2)
        person_priors = torch.tensor(data['person_priors'].numpy()).float().cuda().permute(0, 3, 1, 2)
        exp_seg = torch.tensor(data['exp_seg'].numpy()).float().cuda().permute(0, 3, 1, 2)

        # data_in = {'gt': gt}
        data_in = {
            'person': person,
            'person_priors': person_priors,
            'inpaint_region': inpaint_region,
            'exp_seg': exp_seg
        }
        ourModel.setInput(data_in)
        ourModel.optimize_parameters()

        if (i+1) % config.viz_steps == 0:
            ret_loss = ourModel.get_current_losses()
            if config.pretrain_network is False:
                print(
                    '[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f, adv: %.4f, mrf: %.4f), D_loss: %.4f'
                    % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_rec'], ret_loss['G_loss_ae'],
                       ret_loss['G_loss_adv'], ret_loss['G_loss_mrf'], ret_loss['D_loss']))
                writer.add_scalar('adv_loss', ret_loss['G_loss_adv'], cnt)
                writer.add_scalar('D_loss', ret_loss['D_loss'], cnt)
                writer.add_scalar('G_mrf_loss', ret_loss['G_loss_mrf'], cnt)
            else:
                print('[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f)'
                      % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_rec'], ret_loss['G_loss_ae']))

            writer.add_scalar('G_loss', ret_loss['G_loss'], cnt)
            writer.add_scalar('reconstruction_loss', ret_loss['G_loss_rec'], cnt)
            writer.add_scalar('autoencoder_loss', ret_loss['G_loss_ae'], cnt)

            images = ourModel.get_current_visuals_tensor()
            im_completed = vutils.make_grid(images['completed'], normalize=True, scale_each=True)
            im_input = vutils.make_grid(images['input'], normalize=True, scale_each=True)
            im_gt = vutils.make_grid(images['gt'], normalize=True, scale_each=True)
            writer.add_image('gt', im_gt, cnt)
            writer.add_image('input', im_input, cnt)
            writer.add_image('completed', im_completed, cnt)
            # if (i+1) % config.train_spe == 0:
            #     print('saving model ..')
            #     ourModel.save_networks(epoch+1)
        cnt += 1
    ourModel.save_networks(epoch+1)

writer.export_scalars_to_json(os.path.join(config.model_folder, 'GMCNN_scalars.json'))
writer.close()
