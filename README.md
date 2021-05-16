# Deraindrop

## Visual Attentive GAN Project
### Network
- Generator

  <img src="https://user-images.githubusercontent.com/74402562/118164824-7eab1980-b45e-11eb-9b0e-beab280f1db1.png" width="50%" height="50%"></img>

  * Attention map Generation Module    

  <img src="https://user-images.githubusercontent.com/74402562/118346899-ef932400-b579-11eb-9f74-18b63db2ff3d.png" width="60%" height="60%"></img>

  * Autoencoder - Raindrop Free Generation Module     

  <img src="https://user-images.githubusercontent.com/74402562/118352291-9e942780-b59b-11eb-85ce-ed0ce45f5511.png" width="80%" height="80%"></img>

    <img src="https://user-images.githubusercontent.com/74402562/118352288-9d62fa80-b59b-11eb-8dbe-3c145074c484.png" width="50%" height="50%"></img>
    <img src="https://user-images.githubusercontent.com/74402562/118352290-9e942780-b59b-11eb-85a6-e55ab8c52ad9.png" width="50%" height="50%"></img>
  
- Discriminator

<img src="https://user-images.githubusercontent.com/74402562/118346902-f6219b80-b579-11eb-8f22-47f8ef25d255.png" width="80%" height="80%"></img>

------------------------------------
### Result
- PSNR : 32.04  , SSIM : 0.9686    
<img src="https://user-images.githubusercontent.com/74402562/118389185-a3241300-b663-11eb-8c2d-49b8c9feba28.png" width="60%" height="60%"></img>

- PSNR : 36.48  , SSIM : 0.9880   
<img src="https://user-images.githubusercontent.com/74402562/118389188-a4554000-b663-11eb-8093-ac3b903f7a3e.png" width="60%" height="60%"></img>

- PSNR : 34.05  , SSIM : 0.9726   
<img src="https://user-images.githubusercontent.com/74402562/118389189-a4edd680-b663-11eb-9816-91934057c594.png" width="60%" height="60%"></img>

- PSNR : 33.43  , SSIM : 0.9644   
<img src="https://user-images.githubusercontent.com/74402562/118389190-a4edd680-b663-11eb-9bfd-7f5da654bfb5.png" width="60%" height="60%"></img>

- PSNR : 31.75  , SSIM : 0.9777   
<img src="https://user-images.githubusercontent.com/74402562/118389191-a5866d00-b663-11eb-9e97-189299e05afa.png" width="60%" height="60%"></img>

- PSNR : 36.46  , SSIM : 0.9839   
<img src="https://user-images.githubusercontent.com/74402562/118389194-a5866d00-b663-11eb-9ac1-318dc7658cdd.png" width="60%" height="60%"></img>

- PSNR : 34.41  , SSIM : 0.9847   
<img src="https://user-images.githubusercontent.com/74402562/118389195-a61f0380-b663-11eb-896d-603288ed0cf5.png" width="60%" height="60%"></img>

- PSNR : 34.86  , SSIM : 0.9808   
<img src="https://user-images.githubusercontent.com/74402562/118389196-a61f0380-b663-11eb-8b8c-a3bfbb94fea2.png" width="60%" height="60%"></img>

---------------------------
### Result of Object Detection
<img src="https://user-images.githubusercontent.com/74402562/117578056-7e8ddf80-b127-11eb-9cb4-a5bca46e6b91.png" width="450px" height="180px"></img>

<img src="https://user-images.githubusercontent.com/74402562/117578057-80f03980-b127-11eb-8fca-39e23c05ed17.png" width="450px" height="180px"></img>

<img src="https://user-images.githubusercontent.com/74402562/117578060-83529380-b127-11eb-9049-0fc7e03f0a3b.png" width="450px" height="300px"></img>
<img src="https://user-images.githubusercontent.com/74402562/117578065-864d8400-b127-11eb-993c-be0e9fe84fb0.png" width="450px" height="300px"></img>

-----------------------------
### References

- LSGAN : https://arxiv.org/pdf/1611.04076.pdf
- SRGAN : https://arxiv.org/pdf/1609.04802.pdf
- RCAN(Image Super-Resolution Using Very Deep Residual Channel Attention Networks) : https://arxiv.org/pdf/1807.02758.pdf
- pix2pix : https://arxiv.org/pdf/1611.07004.pdf
- Neural style transfer : https://arxiv.org/pdf/1508.06576v2.pdf
- Loss Functions for Image Restoration with Neural Networks : https://arxiv.org/pdf/1511.08861.pdf
