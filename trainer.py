import os
import torch
from torch import optim
from utils.loss import FocalLoss, SSIM
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import cv2
import random
import importlib
import yaml



Model_type = {
    '1layerRGB':'model.easynet_1layer_single',
    '1layerDepth':'model.easynet_1layer_single',
    '1layerRGBD':'model.easynet_1layer_mul',
    '1layerFusion1':'easynet_1layer_Fusion1',
    '1layerFusion2':'easynet_1layer_Fusion2',

    '2layerRGB':'model.easynet_2layer_single',
    '2layerDepth':'model.easynet_2layer_single',
    '2layerRGBD':'model.easynet_2layer_mul',
    '2layerFusion1':'model.easynet_2layer_Fusion1',
    '2layerFusion2':'model.easynet_2layer_Fusion2',

    '3layerRGB':'model.easynet_3layer_single',
    '3layerDepth':'model.easynet_3layer_single',
    '3layerRGBD':'model.easynet_3layer_mul',
    '3layerFusion1':'model.easynet_3layer_Fusion1',
    '3layerFusion2':'model.easynet_3layer_Fusion2',
}
    
def calc_feature_map_entropy(feature_map):
    flat_feature_map = feature_map.view(-1)
    hist = torch.histc(flat_feature_map, bins=256, min=0, max=1)
    prob = hist / hist.sum()
    entropy = (-prob * torch.log2(prob + 1e-12)).sum()
    return entropy



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def create_file(args):
    if not os.path.exists(args.weight_save_path):
        os.makedirs(args.weight_save_path)
    if not os.path.exists(os.path.join(args.weight_save_path, args.layer_size + '_' + args.mode_type)):
        os.makedirs(os.path.join(args.weight_save_path, args.layer_size + '_' + args.mode_type))

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(os.path.join(args.log_path, args.layer_size + '_' + args.mode_type)):
        os.makedirs(os.path.join(args.log_path, args.layer_size + '_' + args.mode_type))
    
    if not os.path.exists(args.pic_path_train):
        os.makedirs(args.pic_path_train)
    if not os.path.exists(os.path.join(args.pic_path_train, args.layer_size + '_' + args.mode_type)):
        os.makedirs(os.path.join(args.pic_path_train, args.layer_size + '_' + args.mode_type))

    if not os.path.exists(args.record_path):
        os.makedirs(args.record_path)
    if not os.path.exists(os.path.join(args.record_path, args.layer_size + '_' + args.mode_type)):
        os.makedirs(os.path.join(args.record_path, args.layer_size + '_' + args.mode_type))
    


def train_on_device(obj_names, args, dataset_checkpoint):

    create_file(args)

    for obj_name in obj_names:
        run_name = "EasyNet_" + args.layer_size + "_" + args.mode_type + "_" + str(args.lr) + '_' + str(args.epochs) + '_bs' + str(args.bs) + "_"+obj_name + '_' + args.mode_type + '_'
        model = EasyNet(in_channels=3, out_channels=3)

        checkpoint_rgb_path =  dataset_checkpoint['checkpoint_rgb'][obj_name]


        if args.pretrain:
            model.load_state_dict(torch.load(os.path.join(args.weight_save_path, args.layer_size + '_' + args.mode_type, run_name + "best.pckl")))
            model.cuda()
        else:
            model.cuda()
            model.apply(weights_init)

        if args.mode_type == "Fusion2":
            module = importlib.import_module(args.Model_type[args.layer_size+'RGB'])
            EasyNet_rgb = getattr(module, 'ReconstructiveSubNetwork')
            model_rgb = EasyNet_rgb(in_channels=3, out_channels=3)
            model_rgb.load_state_dict(torch.load(checkpoint_rgb_path,map_location=torch.device('cuda:0')),strict=False)
            model_rgb.cuda()
            model_rgb.eval()


        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        img_dim = 256
        train_loader,_ = get_data_loader(args,"train", class_name=obj_name, img_size=(256,256),batch_size=args.bs, num_workers=16,shuffle=True)
        test_loader,datas_len = get_data_loader(args,"test", class_name=obj_name,img_size=[img_dim, img_dim],batch_size=1, num_workers=4,shuffle=False)

        n_iter = 0

        for epoch in range(args.start_epoch, args.epochs):
            loss_all = 0

            # Train
            model.train()
            for i_batch, sample_batched in enumerate(train_loader):      
                gray_aug_image = sample_batched["augmented_image"].cuda()
                gray_aug_depth = sample_batched["augmented_zzz"].cuda()
                gray_image = sample_batched["image"].cuda()
                gray_depth = sample_batched["zzz"].cuda()
                mask = sample_batched["mask"].cuda()

                if args.mode_type == 'RGB':
                    gray_rec_image, out_mask, _ = model(gray_aug_image)
                    out_mask_sm = torch.softmax(out_mask, dim=1)   # Predicted Mask

                    l2_loss_rgb = loss_l2(gray_rec_image,gray_image)  #rgb MAE loss
                    ssim_loss_rgb = loss_ssim(gray_rec_image, gray_image) #rgb ssim loss                      
                    focal_loss = loss_focal(out_mask_sm, mask) #focal loss
                    
                    loss = l2_loss_rgb + ssim_loss_rgb + focal_loss

                elif args.mode_type == 'Depth':
                    gray_rec_depth, out_mask, _ = model(gray_aug_depth)
                    out_mask_sm = torch.softmax(out_mask, dim=1)   # Predicted Mask

                    l2_loss_depth = loss_l2(gray_rec_depth,gray_depth)  #depth MAE loss                    
                    focal_loss = loss_focal(out_mask_sm, mask) #focal loss

                    loss = l2_loss_depth + focal_loss

                elif args.mode_type == 'RGBD':
                    gray_rec_image,gray_rec_depth,out_mask = model(gray_aug_image,gray_aug_depth)
                    out_mask_sm = torch.softmax(out_mask, dim=1)   # Predicted Mask

                    l2_loss_rgb = loss_l2(gray_rec_image,gray_image)  #rgb MAE loss
                    l2_loss_depth = loss_l2(gray_rec_depth,gray_depth)  #depth MAE loss
                    ssim_loss_rgb = loss_ssim(gray_rec_image, gray_image) #rgb ssim loss                      
                    focal_loss = loss_focal(out_mask_sm, mask) #focal loss

                    loss = l2_loss_rgb + ssim_loss_rgb +  l2_loss_depth + focal_loss
                elif args.mode_type == 'Fusion1':
                    gray_rec_image,gray_rec_depth,out_mask,out_mask_rgb,entropy_rgb,entropy_fusion = model(gray_aug_image,gray_aug_depth)
                    out_mask_sm = torch.softmax(out_mask, dim=1)   # Predicted Mask
                    out_mask_sm = torch.softmax(out_mask, dim=1)
                    out_mask_sm_rgb = torch.softmax(out_mask_rgb, dim=1)


                    l2_loss_rgb = loss_l2(gray_rec_image,gray_image)  #rgb MAE loss
                    l2_loss_depth = loss_l2(gray_rec_depth,gray_depth)  #depth MAE loss
                    ssim_loss_rgb = loss_ssim(gray_rec_image, gray_image) #rgb ssim loss                      
                    focal_loss_rgb = loss_focal(out_mask_sm_rgb, mask) #focal loss
                    focal_loss = loss_focal(out_mask_sm, mask) #focal loss

                    loss = l2_loss_rgb + ssim_loss_rgb +  l2_loss_depth + focal_loss_rgb + focal_loss
                elif args.mode_type == 'Fusion2':
                    with torch.no_grad():
                        gray_rec_image,out_mask_rgb, merge_rgb = model_rgb(gray_aug_image)
                    gray_rec_depth,out_mask,entropy_rgb,entropy_fusion = model(gray_aug_depth,merge_rgb)

                    out_mask_sm = torch.softmax(out_mask, dim=1)   # Predicted Mask
                    l2_loss_depth = loss_l2(gray_rec_depth,gray_depth)  #depth MAE loss                    
                    focal_loss = loss_focal(out_mask_sm, mask) #focal loss

                    loss = l2_loss_depth + focal_loss
                

                loss_all += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                n_iter += 1
            scheduler.step()
            # save pth file     
            if (epoch + 1) % 5 == 0: 
                torch.save(model.state_dict(), os.path.join(args.weight_save_path, args.layer_size + '_' + args.mode_type, run_name+".pckl"))
            # print("Epoch: "+str(epoch)+"  loss:"+str(loss_all))
            
            # initialization
            total_pixel_scores = np.zeros((img_dim * img_dim * datas_len))
            total_gt_pixel_scores = np.zeros((img_dim * img_dim * datas_len))
            mask_cnt = 0
            anomaly_score_gt = []
            anomaly_score_prediction = []

            # Evaluation
            model.eval()
            for i_batch, sample_batched in enumerate(test_loader):
    
                gray_image = sample_batched["image"].cuda()
                gray_depth = sample_batched["zzz"].cuda()

                is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
                anomaly_score_gt.append(is_normal)
                true_mask = sample_batched["mask"]
                true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

                if args.mode_type == 'RGB':
                    gray_rec_image,out_mask,_ = model(gray_image)
                elif args.mode_type == 'Depth':
                    gray_rec_depth,out_mask,_ = model(gray_depth)
                elif args.mode_type == 'RGBD':
                    gray_rec_image,gray_rec_depth,out_mask = model(gray_image,gray_depth)
                if args.mode_type == "Fusion1":
                    with torch.no_grad():
                        gray_rec_image,gray_rec_depth,out_mask,out_mask_rgb,entropy_rgb,entropy_fusion = model(gray_image,gray_depth)
                    cfme_rgb = calc_feature_map_entropy(entropy_rgb)
                    cfme_fusion = calc_feature_map_entropy(entropy_fusion)

                    if cfme_fusion >= cfme_rgb:
                        out_mask_sm = torch.softmax(out_mask, dim=1)
                    else:
                        out_mask_sm = torch.softmax(out_mask_rgb, dim=1)
                if args.mode_type == "Fusion2":
                    gray_rec_image,out_mask_rgb, merge_rgb = model_rgb(gray_image)
                    gray_rec_depth,out_mask,entropy_rgb,entropy_fusion = model(gray_depth,merge_rgb)
                    cfme_rgb = calc_feature_map_entropy(entropy_rgb)
                    cfme_fusion = calc_feature_map_entropy(entropy_fusion)
                    out_mask_sm = torch.softmax(out_mask, dim=1)

                    if cfme_fusion >= cfme_rgb:
                        out_mask_sm = torch.softmax(out_mask, dim=1)
                    else:
                        out_mask_sm = torch.softmax(out_mask_rgb, dim=1)

                out_mask_sm = torch.softmax(out_mask, dim=1)
                # print("out_mask:",out_mask.shape)
                if args.save_picture:
                    if i_batch == 0:
                        img_rgb = gray_image.detach().cpu().numpy()*255.0
                        img_depth = gray_depth.detach().cpu().numpy()*255.0
                        img_mask = true_mask_cv*255.0
                        img_out_mask = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()*255.0
                        
                        img_rgb = img_rgb.astype(np.uint8)[0]
                        img_rgb = np.transpose(img_rgb,(1,2,0))
                        img_depth = img_depth.astype(np.uint8)[0]
                        img_depth = np.transpose(img_depth,(1,2,0))
                        img_mask = np.repeat(img_mask,3,axis=2)
                        img_out_mask = np.expand_dims(img_out_mask, axis=2)
                        img_out_mask = np.repeat(img_out_mask,3,axis=2)
                        img_and = img_rgb*(1-img_mask)

                        if args.mode_type == 'RGB':
                            img_rec_image = gray_rec_image.detach().cpu().numpy()*255.0
                            img_rec_image = img_rec_image.astype(np.uint8)[0]
                            img_rec_image = np.transpose(img_rec_image,(1,2,0))
                            concate_img = np.concatenate((img_rgb,img_rec_image,img_and,img_mask,img_out_mask),axis=1)
                        if args.mode_type == 'Depth':
                            img_rec_depth = gray_rec_depth.detach().cpu().numpy()*255.0
                            img_rec_depth = img_rec_depth.astype(np.uint8)[0]
                            img_rec_depth = np.transpose(img_rec_depth,(1,2,0))
                            concate_img = np.concatenate((img_depth,img_rec_depth,img_mask,img_out_mask),axis=1)
                        # elif args.mode_type == 'RGBD':
                        elif (args.mode_type == 'RGBD') or ('Fusion' in args.mode_type):
                            img_rec_image = gray_rec_image.detach().cpu().numpy()*255.0
                            img_rec_image = img_rec_image.astype(np.uint8)[0]
                            img_rec_image = np.transpose(img_rec_image,(1,2,0))
                            img_rec_depth = gray_rec_image.detach().cpu().numpy()*255.0
                            img_rec_depth = img_rec_depth.astype(np.uint8)[0]
                            img_rec_depth = np.transpose(img_rec_depth,(1,2,0))
                            concate_img = np.concatenate((img_rgb,img_rec_image,img_and,img_depth,img_rec_depth,img_mask,img_out_mask),axis=1)

                        cv2.imwrite(os.path.join(args.pic_path_train, args.layer_size + '_' + args.mode_type, obj_name+str(epoch)+".png"), concate_img)

                out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
                out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                                padding=21 // 2).cpu().detach().numpy()
                image_score = np.max(out_mask_averaged)
                anomaly_score_prediction.append(image_score)

                flat_true_mask = true_mask_cv.flatten()
                flat_out_mask = out_mask_cv.flatten()
                total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
                total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
                mask_cnt += 1

            anomaly_score_prediction = np.array(anomaly_score_prediction)
            anomaly_score_gt = np.array(anomaly_score_gt)
            auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
            ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

            total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
            total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
            total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
            auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
            ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

            if epoch == args.start_epoch:
                auroc_old = auroc
                auroc_pixel_old = auroc_pixel
                best_epoch = epoch

            with open(os.path.join(args.record_path, args.layer_size + '_' + args.mode_type, obj_name+"run.txt"), "a") as filewrite:
                filewrite.write("Epoch: "+str(epoch)+"  loss:"+str(loss_all)+"AUC Image:  " +str(auroc)+"AP Image:  " +str(ap)+"AUC Pixel:  " +str(auroc_pixel)+"AP Pixel:  " +str(ap_pixel)+"\n")
            if epoch % 50 == 0:
                print("Epoch: "+str(epoch)+"  loss:"+str(loss_all)+"AUC Image:  " +str(auroc)+"AP Image:  " +str(ap)+"AUC Pixel:  " +str(auroc_pixel)+"AP Pixel:  " +str(ap_pixel))
            
            if (epoch > 200) and (auroc_old < auroc) or (auroc_old == auroc and auroc_pixel_old < auroc_pixel):
                # use early stop，for there are many loss targets (ssim loss、focal loss and mse loss) to converge, the convergence curve fluctuates greatly.
                torch.save(model.state_dict(), os.path.join(args.weight_save_path, args.layer_size + '_' + args.mode_type, run_name+"best.pckl"))
                auroc_old = auroc
                auroc_pixel_old = auroc_pixel
                best_epoch = epoch
                print("==============================")
                print(obj_name+"'s best epoch: "+str(best_epoch))
                print("AUC Image:  " +str(auroc))
                print("AP Image:  " +str(ap))
                print("AUC Pixel:  " +str(auroc_pixel))
                print("AP Pixel:  " +str(ap_pixel))
                print("==============================")
                with open(os.path.join(args.record_path, args.layer_size + '_' + args.mode_type, obj_name+"best.txt"), "a") as filewrite:
                    filewrite.write(obj_name+"'s best epoch: "+str(best_epoch)+"\n")
                    filewrite.write("AUC Image:  " +str(auroc)+"\n")
                    filewrite.write("AP Image:  " +str(ap)+"\n")
                    filewrite.write("AUC Pixel:  " +str(auroc_pixel)+"\n")
                    filewrite.write("AP Pixel:  " +str(ap_pixel)+"\n")
                    filewrite.write("==============================\n")

    
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', type=int, nargs='+',required=True)
    parser.add_argument('--bs', type=int, default=4, required=False)
    parser.add_argument('--lr', type=float, default=0.0002, required=False)
    parser.add_argument('--seed', type=int, default=66, required=False)
    parser.add_argument('--start_epoch', type=int, default = 0, required=False)
    parser.add_argument('--epochs', type=int, default=1000, required=False)
    parser.add_argument('--gpu_id', type=int, default=7, required=False)

    parser.add_argument('--pretrain', action='store_true', help='Save the visual data')
    
    parser.add_argument('--dataset_type', default='Mvtec3D_AD', type=str, 
                        choices=['Mvtec3D_AD','Eyecandies'],help='Choose Mvtec 3D AD or Eyecandies dataset! ')

    parser.add_argument('--layer_size', default='2layer', type=str, 
                        choices=['1layer','2layer','3layer'],
                        help='Select the number of layers of the network! ')
    parser.add_argument('--mode_type', default='RGBD', type=str, 
                        choices=['RGB','Depth',"RGBD","Fusion1","Fusion2"],help='Choose mode type to train! ')    

    parser.add_argument('--save_picture', action='store_true', help='Save the visual data')

    parser.add_argument('--checkpoint_yaml', type=str, default="./checkpoint/checkpoint.yaml", required=False)


    args = parser.parse_args()

    with open(args.checkpoint_yaml, 'r') as file:
        yaml_data = yaml.safe_load(file)
    args_dict = vars(args)
    args_dict.update(yaml_data)
    args = argparse.Namespace(**args_dict)

    setup_seed(args.seed)
    picked_classes = []
    try:
        module_name = args.Model_type[args.layer_size+args.mode_type]
    except KeyError:
        raise KeyError(f"model network '{args.model_type}' does not support in the project.")
    
    # 
    module = importlib.import_module(module_name)

# 
    EasyNet = getattr(module, 'ReconstructiveSubNetwork')
    if args.dataset_type == 'Mvtec3D_AD':
        from data.mvtec3d_dataset import get_data_loader, mvtec3d_classes
        dataset_checkpoint = args.mvted3dad
        obj_batch = mvtec3d_classes()
    elif args.dataset_type == 'Eyecandies':
        from data.eyecandies_dataset import get_data_loader, eyecandies_classes
        dataset_checkpoint = args.eyecandies
        obj_batch = eyecandies_classes()

    if int(args.obj_id[0]) == -1:
        picked_classes = obj_batch
    else:
        for i in args.obj_id:
            picked_classes.append(obj_batch[int(i)])

    print('class ', picked_classes, ' will be trained!')
    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args, dataset_checkpoint)


#  python trainer.py --gpu_id 0 --obj_id -1 --layer_size 1layer --epochs 5 --lr 0.0002 --mode_type RGB
