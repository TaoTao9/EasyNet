import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weight(m):
    
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

class ReconstructiveSubNetwork(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, base_width=128,img_size=256,train_model=True):
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.encoder_d = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)
        self.decoder_d = DecoderReconstructive(base_width, out_channels=out_channels)
        self.hidden_size = 512
        self.hidden_size1 = 256
        self.hidden_size2 = 2
        self.img_size = img_size

        self.mlp1 = nn.Sequential(
            nn.Conv2d(base_width*2,self.hidden_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(0.2)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(base_width*3,self.hidden_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(0.2)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv2d(base_width*6,self.hidden_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(0.2)
        )


        self.mlp1_d = nn.Sequential(
            nn.Conv2d(base_width*2,self.hidden_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(0.2)
        )
        self.mlp2_d = nn.Sequential(
            nn.Conv2d(base_width*3,self.hidden_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(0.2)
        )
        self.mlp3_d = nn.Sequential(
            nn.Conv2d(base_width*6,self.hidden_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(0.2)
        )
        self.mlp4 = nn.Sequential(
            nn.Conv2d(self.hidden_size*6,self.hidden_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.hidden_size,self.hidden_size1, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_size1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.hidden_size1,self.hidden_size2, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_size2),
            nn.LeakyReLU(0.2)
        )
        if train_model:
            self.apply(init_weight)

    def forward(self, x, d):
        b1,b2,b3,b5 = self.encoder(x)
        d1,d2,d3,d5 = self.encoder_d(d)

        output,db2,db3,db4 = self.decoder(b5)        
        output_d,dd2,dd3,dd4 = self.decoder_d(d5)   

        merge1 = self.mlp1(torch.cat((b1,db4), dim=1))
        merge2 = self.mlp2(torch.cat((b2,db3), dim=1))
        merge3 = self.mlp3(torch.cat((b3,db2), dim=1))
        merge2 = torch.nn.functional.interpolate(merge2, size=(self.img_size,self.img_size), mode='bilinear', align_corners=False)
        merge3 = torch.nn.functional.interpolate(merge3, size=(self.img_size,self.img_size), mode='bilinear', align_corners=False)

        merge_d1 = self.mlp1_d(torch.cat((d1,dd4), dim=1))
        merge_d2 = self.mlp2_d(torch.cat((d2,dd3), dim=1))
        merge_d3 = self.mlp3_d(torch.cat((d3,dd2), dim=1))
        merge_d2 = torch.nn.functional.interpolate(merge_d2, size=(self.img_size,self.img_size), mode='bilinear', align_corners=False)
        merge_d3 = torch.nn.functional.interpolate(merge_d3, size=(self.img_size,self.img_size), mode='bilinear', align_corners=False)
        
        mask = self.mlp4(torch.cat((merge1,merge2,merge3,merge_d1,merge_d2,merge_d3), dim=1))
        return output, output_d, mask


class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        return b1,b2,b3,b5

class DecoderReconstructive(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderReconstructive, self).__init__()

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width*2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width*2),
                                 nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))
        #self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b5):
        up1 = self.up1(b5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        out = self.fin_out(db4)
        return out,db2,db3,db4

        
