import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pylab
from skimage.measure import compare_psnr, compare_ssim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

lr = 0.0002

def torch_variable(x,is_train):
	if is_train:
		return Variable(torch.from_numpy(np.array(x).transpose((0,3,1,2))),requires_grad=True).to(device)
	else:
		with torch.no_grad():
			result = torch.from_numpy(np.array(x).transpose((0,3,1,2))).to(device)
		return result
  
def get_mask(dg_img,img):
	# downgraded image - image
	mask = np.fabs(dg_img-img)
	# threshold under 25
	mask[np.where(mask<(25.0/255.0))] = 0.0
	mask[np.where(mask>0.0)] = 1.0
	mask = np.max(mask, axis=2)
	mask = np.expand_dims(mask, axis=2)
	
	return mask

def calc_psnr(im1, im2):
	im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
	im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
	
	return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
	im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
	im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
	
	return compare_ssim(im1_y, im2_y)

def align_to_four(img):
	a_row = int(img.shape[0]/4)*4
	a_col = int(img.shape[1]/4)*4
	img = img[0:a_row,0:a_col]
	
	return img

def predict(image, model):
	image = image.transpose((2,0,1))
  	image = image[np.newaxis,:,:,:]
  	image = torch.from_numpy(image)
  	image = Variable(image).to(device)
  	out = model(image)[-1]
  	out = out.cpu().data
  	out = out.numpy()
  	out = out.transpose((0,2,3,1))
  	out = out[0,:,:,:]*255.
	
	return out

def minmax_scale(input_arr):
	"""
    	:param input_arr:
    	:return:
    	"""
    	min_val = np.min(input_arr)
    	max_val = np.max(input_arr)
    	output_arr = (input_arr - min_val) / (max_val - min_val)

    	return output_arr
  
class trainer:
	def __init__(self, iter=500, batch_size=1):
		self.net_D = Discriminator().to(device)
    		self.net_G = Generator().to(device)
    		self.optim_D = torch.optim.Adam(filter(lambda p : p.requires_grad, self.net_D.parameters()), lr = lr, betas = (0.5,0.999))
    		self.optim_G = torch.optim.Adam(filter(lambda p : p.requires_grad, self.net_G.parameters()), lr = lr, betas = (0.5,0.999))
    		self.iter = iter
    		self.batch_size = batch_size
    		self.expr_dir = './checkpoints'

    		# Attention Loss
    		self.criterionAtt = AttentionLoss(theta=0.9, iteration=4)  
    		# GAN Loss
    		self.criterionGAN = GANLoss(real_label=1.0, fake_label=0.0)
   		 # Perceptual Loss
    		self.criterionPL = PerceptualLoss()
   		 # Multiscale Loss
    		self.criterionML = MultiscaleLoss(batch=self.batch_size) 
    		# Mask Loss
    		self.criterionMask = MaskLoss()
    		# MAP Loss
    		self.criterionMAP = MAPLoss()

    		self.out_path = './weight/'
		
  	def forward_process(self, I, GT):
		M_ = []
    		for i in range(I.shape[0]):
			M_.append(get_mask(np.array(I[i]),np.array(GT[i])))
    		M_ = np.array(M_)
    		M_ = torch_variable(M_, False)
    		I_ = torch_variable(I, False)
    		GT_= torch_variable(GT, False)

    		A_, t1, t2, out = self.net_G(I_)
    		S_ = [t1,t2,out]
		
		return A_, I_, GT_, M_, S_, out

  	def train_start(self):
    		# I_ : input raindrop image
    		# A_ : attention map
    		# M_ : mask GT
    		# out : output image of the autoencoder
    		# GT_ : GT

    		for epoch in range(1, self.iter+1):
			tot_loss_G = 0.0
			tot_loss_D = 0.0
      			tot_loss_PL = 0.0
      			tot_loss_ML = 0.0
      			tot_loss_att = 0.0
      			tot_loss_MAP = 0.0
      			tot_loss_GAN = 0.0
      			tot_loss_disc = 0.0
      			tot_loss_mask = 0.0
      			count = 0

      			for i, (I_,GT_) in enumerate(train_loader):
				count+=1
				A_, I_, GT_, M_, S_, out = self.forward_process(I, GT)
				
				## train D ##
				
				self.optim_D.zero_grad()
				
				D_map_R, D_real = self.net_D(GT_)
        			D_map_O, D_fake = self.net_D(out.detach())
        
        			loss_MAP = self.criterionMAP(D_map_O, D_map_R, I.detach(), GT.detach())

        			loss_fake = self.criterionGAN(D_fake,is_real=False)
        			loss_real = self.criterionGAN(D_real,is_real=True)
				
        			loss_disc = 0.5 * loss_real + 0.5 * loss_fake
				
        			loss_D = loss_disc + loss_MAP
        			loss_D.backward() 
        			self.optim_D.step()
				
				## train G ##
				
        			self.optim_G.zero_grad()
				
				_ , fake = self.net_D(out)
				
        			loss_att = self.criterionAtt(A_,M_.detach())

        			loss_PL = self.criterionPL(out, GT_)
        
        			loss_ML = self.criterionML(S_, GT.detach())

        			loss_Mask = self.criterionMask(out, GT_.detach(), M_.detach())

        			loss_gan = self.criterionGAN(fake,is_real=True)

        			loss_G = 0.05 * loss_gan + loss_att + 2 * loss_ML + 0.7 * loss_PL + 10 * loss_Mask
        			loss_G.backward()      
        			self.optim_G.step()

        			tot_loss_G += loss_G.item()
        			tot_loss_D += loss_D.item()
        			tot_loss_PL += loss_PL.item()
        			tot_loss_ML += loss_ML.item()
        			tot_loss_att += loss_att.item()
        			tot_loss_MAP += loss_MAP.item()
        			tot_loss_GAN += loss_gan.item()
        			tot_loss_disc += loss_disc.item()
        			tot_loss_mask += loss_Mask.item()

        			if count == 1:
         				print('count: {},loss_G: {:.4f} ,loss_D: {:.4f} ,loss_GAN: {:.4f} ,loss_disc: {:.4f} , loss_PL: {:.4f} ,loss_ML: {:.4f} ,loss_Att: {:.4f} ,loss_MAP: {:.4f} , loss_Mask: {:.4f}'.format(
																								count,tot_loss_G,tot_loss_D,
                                                                                                                                                                                                tot_loss_GAN,tot_loss_disc,
                                                                                                                                                                                                tot_loss_PL,tot_loss_ML,tot_loss_att,
                                                                                                                                                                                                tot_loss_MAP,tot_loss_mask))

        			if count % 20 == 0:
          				print('count: {},loss_G: {:.4f} ,loss_D: {:.4f} ,loss_GAN: {:.4f} ,loss_disc: {:.4f}  ,loss_PL: {:.4f} ,loss_ML: {:.4f} ,loss_Att: {:.4f} ,loss_MAP: {:.4f} , loss_Mask: {:.4f}'.format(
																						count,tot_loss_G/20,tot_loss_D/20,
                                                                                                                                                                             	tot_loss_GAN/20,tot_loss_disc/20,
                                                                                                                                                                             	tot_loss_PL/20,tot_loss_ML/20,tot_loss_att/20,
                                                                                                                                                                              	tot_loss_MAP/20,tot_loss_mask/20))
          				tot_loss_G = 0.0
         	 			tot_loss_D = 0.0
          				tot_loss_PL = 0.0
          				tot_loss_ML = 0.0
          				tot_loss_att = 0.0
          				tot_loss_MAP = 0.0
          				tot_loss_GAN = 0.0
          				tot_loss_disc = 0.0
          				tot_loss_mask = 0.0

      				step = 0
      				cumulative_psnr=0
      				cumulative_ssim=0

     				with torch.no_grad():
					for i, (I_val,GT_val) in enumerate(valid_dataset):
          					img=align_to_four(I_val)
          					GT_val=align_to_four(GT_val)
          					result=predict(img, self.net_G)
          					result=minmax_scale(result)
          					cumulative_psnr+=calc_psnr(result,GT_val)
          					cumulative_ssim+=calc_ssim(result,GT_val)
          					step+=1
          					if epoch % 20 == 0 and step % 10 == 0:
            					plt.figure(figsize=(10, 10))
            					plt.subplot(1,3,1)
						plt.imshow(I_val)
						plt.axis('off')
						plt.subplot(1,3,2)
						plt.imshow(result)
						plt.axis('off')
						plt.subplot(1,3,3)
						plt.imshow(GT_val)
						plt.axis('off')
						plt.show()        
      
        				print('Epoch : %d , In validation dataset, PSNR is %.4f and SSIM is %.4f'%(epoch, cumulative_psnr/step, cumulative_ssim/step))

      				if not os.path.exists(self.out_path):
        				os.system('mkdir -p {}'.format(self.out_path))
      				w_name = 'G_epoch_{}_PSNR_{:.2f}_SSIM_{:.4f}.pth'.format(epoch,cumulative_psnr/step,cumulative_ssim/step)
      				save_path = os.path.join(self.out_path, w_name)
      				torch.save(self.net_G.state_dict(), save_path)

      				if epoch % 30 == 0 or epoch % 50 == 0 :
        				torch.save(self.net_D.state_dict(),"net_D_{}.pth".format(epoch))

    				return A_, I_, M, D_map_O, D_map_R, out
