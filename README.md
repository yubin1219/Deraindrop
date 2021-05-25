# Deraindrop

## Visual Attentive GAN Project
### Network
- Generator

  <img src="https://user-images.githubusercontent.com/74402562/118164824-7eab1980-b45e-11eb-9b0e-beab280f1db1.png" width="50%" height="50%"></img>

  * Attention map Generation Module    

  <img src="https://user-images.githubusercontent.com/74402562/118443035-61a86c00-b726-11eb-93aa-389e755347d1.png" width="60%" height="60%"></img>
  <img src="https://user-images.githubusercontent.com/74402562/119268728-3692b080-bc2f-11eb-83fd-4f371cd3eba0.png" width="100%" height="100%"></img>

  * Autoencoder - Raindrop Free Generation Module     

  <img src="https://user-images.githubusercontent.com/74402562/118352291-9e942780-b59b-11eb-85ce-ed0ce45f5511.png" width="80%" height="80%"></img>

    <img src="https://user-images.githubusercontent.com/74402562/118352288-9d62fa80-b59b-11eb-8dbe-3c145074c484.png" width="50%" height="50%"></img>
    <img src="https://user-images.githubusercontent.com/74402562/118352290-9e942780-b59b-11eb-85a6-e55ab8c52ad9.png" width="50%" height="50%"></img>
  
- Discriminator

<img src="https://user-images.githubusercontent.com/74402562/118443040-62410280-b726-11eb-9a98-e1587be946ff.png" width="80%" height="80%"></img>
<img src="https://user-images.githubusercontent.com/74402562/118443043-640ac600-b726-11eb-9823-bea25a943c68.png" width="30%" height="30%"></img>

------------------------------------
### Result

- PSNR : 34.83  , SSIM : 0.9788    
<img src="https://user-images.githubusercontent.com/74402562/119271200-81b2c080-bc3b-11eb-9bbc-b313a3dfb37a.png" width="80%" height="80%"></img>

- PSNR : 30.88  , SSIM : 0.9588   
<img src="https://user-images.githubusercontent.com/74402562/119271210-87a8a180-bc3b-11eb-834b-014dc783e70b.png" width="80%" height="80%"></img>

- PSNR : 31.78  , SSIM : 0.9613   
<img src="https://user-images.githubusercontent.com/74402562/119271207-86777480-bc3b-11eb-9a86-c108302ebc49.png" width="80%" height="80%"></img>

- PSNR : 33.42  , SSIM : 0.9680   
<img src="https://user-images.githubusercontent.com/74402562/119271212-88413800-bc3b-11eb-98a5-231471d2c1d9.png" width="80%" height="80%"></img>

- PSNR : 33.59  , SSIM : 0.9697   
<img src="https://user-images.githubusercontent.com/74402562/119271209-87a8a180-bc3b-11eb-9d35-19114b758865.png" width="80%" height="80%"></img>

- PSNR : 35.36  , SSIM : 0.9792   
<img src="https://user-images.githubusercontent.com/74402562/119271211-88413800-bc3b-11eb-987a-77c058f0aba5.png" width="80%" height="80%"></img>

---------------------------
### Result of Object Detection
<img src="https://user-images.githubusercontent.com/74402562/117578056-7e8ddf80-b127-11eb-9cb4-a5bca46e6b91.png" width="40%" height="40%"></img>

<img src="https://user-images.githubusercontent.com/74402562/117578057-80f03980-b127-11eb-8fca-39e23c05ed17.png" width="40%" height="40%"></img>

-----------------------------
### References

- LSGAN : https://arxiv.org/pdf/1611.04076.pdf
- SRGAN : https://arxiv.org/pdf/1609.04802.pdf
- RCAN(Image Super-Resolution Using Very Deep Residual Channel Attention Networks) : https://arxiv.org/pdf/1807.02758.pdf
- pix2pix : https://arxiv.org/pdf/1611.07004.pdf
- Neural style transfer : https://arxiv.org/pdf/1508.06576v2.pdf
- Loss Functions for Image Restoration with Neural Networks : https://arxiv.org/pdf/1511.08861.pdf
