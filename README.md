# Deraindrop

## Visual Attentive GAN Project
### Network
- Generator

  <img src="https://user-images.githubusercontent.com/74402562/118164824-7eab1980-b45e-11eb-9b0e-beab280f1db1.png" width="50%" height="50%"></img>

  * Attention map Generation Module    

  <img src="https://user-images.githubusercontent.com/74402562/118443035-61a86c00-b726-11eb-93aa-389e755347d1.png" width="60%" height="60%"></img>

  * Autoencoder - Raindrop Free Generation Module     

  <img src="https://user-images.githubusercontent.com/74402562/118352291-9e942780-b59b-11eb-85ce-ed0ce45f5511.png" width="80%" height="80%"></img>

    <img src="https://user-images.githubusercontent.com/74402562/118352288-9d62fa80-b59b-11eb-8dbe-3c145074c484.png" width="50%" height="50%"></img>
    <img src="https://user-images.githubusercontent.com/74402562/118352290-9e942780-b59b-11eb-85a6-e55ab8c52ad9.png" width="50%" height="50%"></img>
  
- Discriminator

<img src="https://user-images.githubusercontent.com/74402562/118443040-62410280-b726-11eb-9a98-e1587be946ff.png" width="80%" height="80%"></img>
<img src="https://user-images.githubusercontent.com/74402562/118443043-640ac600-b726-11eb-9823-bea25a943c68.png" width="30%" height="30%"></img>

------------------------------------
### Result

- PSNR : 34.62  , SSIM : 0.9781    
<img src="https://user-images.githubusercontent.com/74402562/119220860-82f2c900-bb27-11eb-8e46-3fbdad6bb157.png" width="60%" height="60%"></img>

- PSNR : 31.11  , SSIM : 0.9586   
<img src="https://user-images.githubusercontent.com/74402562/119220864-86865000-bb27-11eb-9d50-8db9684e8b76.png" width="60%" height="60%"></img>

- PSNR : 31.43  , SSIM : 0.9623   
<img src="https://user-images.githubusercontent.com/74402562/119220866-86865000-bb27-11eb-960c-f2e0a2494090.png" width="60%" height="60%"></img>

- PSNR : 33.15  , SSIM : 0.9667   
<img src="https://user-images.githubusercontent.com/74402562/119220871-871ee680-bb27-11eb-93cd-e50f305cdc93.png" width="60%" height="60%"></img>

- PSNR : 33.41  , SSIM : 0.9689   
<img src="https://user-images.githubusercontent.com/74402562/119220872-871ee680-bb27-11eb-8219-2995cdc87725.png" width="60%" height="60%"></img>

- PSNR : 35.23  , SSIM : 0.9785   
<img src="https://user-images.githubusercontent.com/74402562/119220873-87b77d00-bb27-11eb-8568-fc4b62ec47b1.png" width="60%" height="60%"></img>

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
