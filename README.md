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

- PSNR : 34.01  , SSIM : 0.9756    
<img src="https://user-images.githubusercontent.com/74402562/119158025-bd5d5700-ba90-11eb-834f-ec3d469c3d91.png" width="60%" height="60%"></img>

- PSNR : 30.96  , SSIM : 0.9546   
<img src="https://user-images.githubusercontent.com/74402562/119158038-bf271a80-ba90-11eb-9122-60516c607ab2.png" width="60%" height="60%"></img>

- PSNR : 31.12  , SSIM : 0.9589   
<img src="https://user-images.githubusercontent.com/74402562/119158032-be8e8400-ba90-11eb-91a1-2467fafa7470.png" width="60%" height="60%"></img>

- PSNR : 32.39  , SSIM : 0.9631   
<img src="https://user-images.githubusercontent.com/74402562/119158040-bf271a80-ba90-11eb-86bb-0a120216b4f2.png" width="60%" height="60%"></img>

- PSNR : 32.74  , SSIM : 0.9649   
<img src="https://user-images.githubusercontent.com/74402562/119158033-be8e8400-ba90-11eb-9d03-83a7e6df7960.png" width="60%" height="60%"></img>

- PSNR : 34.60  , SSIM : 0.9762   
<img src="https://user-images.githubusercontent.com/74402562/119158043-bfbfb100-ba90-11eb-9f5f-cea72b727fea.png" width="60%" height="60%"></img>

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
