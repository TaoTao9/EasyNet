import os
import torch
from data.mvtec3d_dataset import get_data_loader
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
import yaml
import importlib



def calc_feature_map_entropy(feature_map):
    flat_feature_map = feature_map.view(-1)
    hist = torch.histc(flat_feature_map, bins=256, min=0, max=1)
    prob = hist / hist.sum()
    entropy = (-prob * torch.log2(prob + 1e-12)).sum() 
    return entropy

def test_once_class(args,obj_name,dataset_checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_dim = 256
    test_loader,datas_len = get_data_loader(args,"test", class_name=obj_name,img_size=[img_dim,img_dim],batch_size=1, num_workers=0,shuffle=False)

    
    checkpoint_rgb_path =  dataset_checkpoint['checkpoint_rgb'][obj_name]
    checkpoint_depth_path =  dataset_checkpoint['checkpoint_depth'][obj_name]
    checkpoint_fusion_path =  dataset_checkpoint['checkpoint_fusion'][obj_name]
    entropy_a =  dataset_checkpoint['entropy'][obj_name]
    # print(entropy_a)

    try:
        module_name = args.Model_type[args.layer_size+args.mode_type]
    except KeyError:
        raise KeyError(f"model network '{args.model_type}' does not support in the project.")
    module = importlib.import_module(module_name)

    if args.mode_type == "RGB":
        checkpoint_path =  checkpoint_rgb_path
    if args.mode_type == "Depth":
        checkpoint_path =  checkpoint_depth_path
    if "Fusion" in args.mode_type:
        checkpoint_path =  checkpoint_fusion_path

    # 获取需要导入的类
    EasyNet = getattr(module, 'ReconstructiveSubNetwork')
    model = EasyNet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cuda:0')),strict=False)
    model = model.to(device)
    model.eval()

    if args.mode_type == "Fusion2":
        module = importlib.import_module(args.Model_type[args.layer_size+'RGB'])
        # module = importlib.import_module(module_name)
        EasyNet_rgb = getattr(module, 'ReconstructiveSubNetwork')
        model_rgb = EasyNet_rgb(in_channels=3, out_channels=3)
        model_rgb.load_state_dict(torch.load(checkpoint_rgb_path,map_location=torch.device('cuda:0')),strict=False)
        model_rgb = model_rgb.to(device)
        model_rgb.eval()

    

    total_pixel_scores = np.zeros((img_dim * img_dim * datas_len))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * datas_len))
    anomaly_score_gt = []
    anomaly_score_prediction = []
    mask_cnt = 0
    predictions = []
    gts = []

    for i_batch, sample_batched in enumerate(tqdm(test_loader)):
        
        gray_image = sample_batched["image"].to(device)
        gray_depth = sample_batched["zzz"].to(device)
        true_mask = sample_batched["mask"]
        is_normal = sample_batched["has_anomaly"].detach().numpy()[0,0]
        anomaly_score_gt.append(is_normal)
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))


        
        if args.mode_type == 'RGB':
            gray_rec_image,out_mask,_ = model(gray_image)
            out_mask_sm = torch.softmax(out_mask, dim=1)
        elif args.mode_type == 'Depth':
            gray_rec_depth,out_mask,_ = model(gray_depth)
            out_mask_sm = torch.softmax(out_mask, dim=1)
        elif args.mode_type == 'RGBD':
            gray_rec_image,gray_rec_depth,out_mask = model(gray_image,gray_depth)
            out_mask_sm = torch.softmax(out_mask, dim=1)
        if args.mode_type == "Fusion1":
            with torch.no_grad():
                gray_rec_image,gray_rec_depth,out_mask,out_mask_rgb,entropy_rgb,entropy_fusion = model(gray_image,gray_depth)
            cfme_rgb = calc_feature_map_entropy(entropy_rgb)
            cfme_fusion = calc_feature_map_entropy(entropy_fusion)

            if cfme_fusion >= cfme_rgb + entropy_a:
                out_mask_sm = torch.softmax(out_mask, dim=1)
            else:
                out_mask_sm = torch.softmax(out_mask_rgb, dim=1)
        if args.mode_type == "Fusion2":
            gray_rec_image,out_mask_rgb, merge_rgb = model_rgb(gray_image)
            gray_rec_depth,out_mask,entropy_rgb,entropy_fusion = model(gray_depth,merge_rgb)
            cfme_rgb = calc_feature_map_entropy(entropy_rgb)
            cfme_fusion = calc_feature_map_entropy(entropy_fusion)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            if cfme_fusion >= cfme_rgb + entropy_a:
                out_mask_sm = torch.softmax(out_mask, dim=1)
            else:
                out_mask_sm = torch.softmax(out_mask_rgb, dim=1)

        if args.save_picture:
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
            elif (args.mode_type == 'RGBD') or ('Fusion' in args.mode_type):
                img_rec_image = gray_rec_image.detach().cpu().numpy()*255.0
                img_rec_image = img_rec_image.astype(np.uint8)[0]
                img_rec_image = np.transpose(img_rec_image,(1,2,0))
                img_rec_depth = gray_rec_image.detach().cpu().numpy()*255.0
                img_rec_depth = img_rec_depth.astype(np.uint8)[0]
                img_rec_depth = np.transpose(img_rec_depth,(1,2,0))
                concate_img = np.concatenate((img_rgb,img_rec_image,img_and,img_depth,img_rec_depth,img_mask,img_out_mask),axis=1)

            cv2.imwrite(os.path.join(args.pic_path_test, args.layer_size + '_' + args.mode_type, obj_name+str(i_batch)+".png"), concate_img)





        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                        padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)
        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask

        predictions.append(out_mask_cv.squeeze())
        gts.append(true_mask_cv.squeeze())

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

    gts = np.array(gts)
    predictions = np.array(predictions)
    aupro,_ = calculate_au_pro(gts,predictions)

    print(("AUC Image:  " +str(auroc)+"AP Image:  " +str(ap)+"AUC Pixel:  " +str(auroc_pixel)+"AP Pixel:  " +str(ap_pixel)+"aupro: "+str(aupro)+"\n"))


def test_on_device(args):
    picked_classes = []

    if args.save_picture:
        if not os.path.exists(args.pic_path_test):
            os.makedirs(args.pic_path_test)
        if not os.path.exists(os.path.join(args.pic_path_test, args.layer_size + '_' + args.mode_type)):
            os.makedirs(os.path.join(args.pic_path_test, args.layer_size + '_' + args.mode_type))

    if args.dataset_type == 'Mvtec3D_AD':
        from data.mvtec3d_dataset import get_data_loader, mvtec3d_classes
        obj_batch = mvtec3d_classes()
        dataset_checkpoint = args.mvted3dad
        
    elif args.dataset_type == 'Eyecandies':
        from data.eyecandies_dataset import get_data_loader, eyecandies_classes
        obj_batch = eyecandies_classes()
        dataset_checkpoint = args.eyecandies


    if int(args.obj_id[0]) == -1:
        picked_classes = obj_batch
    else:
        for i in args.obj_id:
            picked_classes.append(obj_batch[int(i)])

    print('class ', picked_classes, ' will be tested!')

    for obj_name in picked_classes:
        test_once_class(args,obj_name,dataset_checkpoint)


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', type=int, nargs='+',required=True)
    parser.add_argument('--gpu_id', type=int, default=0, required=False)
    parser.add_argument('--dataset_type', default='Mvtec3D_AD', type=str, 
                        choices=['Mvtec3D_AD','Eyecandies'],help='Choose Mvtec 3D AD or Eyecandies dataset! ')
    parser.add_argument('--checkpoint_yaml', type=str, default="/home/yangzesheng/tao_workspace/easynet_union/EasyNet/checkpoint/checkpoint.yaml", required=False)

    parser.add_argument('--save_picture', action='store_true', help='Save the visual data')

    parser.add_argument('--layer_size', default='2layer', type=str, 
                        choices=['1layer','2layer','3layer'],
                        help='Select the number of layers of the network! ')
    parser.add_argument('--mode_type', default='Fusion1', type=str, 
                        choices=['RGB','Depth',"RGBD","Fusion1","Fusion2"],help='Choose mode type to train! ')
    


    args = parser.parse_args()
    
    with open(args.checkpoint_yaml, 'r') as file:
        yaml_data = yaml.safe_load(file)
    args_dict = vars(args)
    args_dict.update(yaml_data)
    args = argparse.Namespace(**args_dict)
    

    with torch.cuda.device(args.gpu_id):
        test_on_device(args)

#python test.py --gpu_id 4 --obj_id -1 --layer_size 2layer --mode_type Fusion
#python test.py --gpu_id 4 --obj_id -1 --layer_size 2layer --mode_type RGB