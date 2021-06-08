# Deraindrop

## Visual Attentive GAN Project
### Network
- Generator

  <img src="https://user-images.githubusercontent.com/74402562/118164824-7eab1980-b45e-11eb-9b0e-beab280f1db1.png" width="50%" height="50%"></img>

  * Attention map Generation Module    

  <img src="https://user-images.githubusercontent.com/74402562/120968059-55528480-c7a3-11eb-8289-cd9c68e42af1.png" width="60%" height="60%"></img>
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
### Comparison with Original Attentive GAN   
- AttGAN : PSNR 26.25 / SSIM 0.9433
- Ours : PSNR 36.34 / SSIM 0.9824    
<img src="https://user-images.githubusercontent.com/74402562/121190059-a5197480-c8a5-11eb-8992-8f543df84ecc.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190059-a5197480-c8a5-11eb-8992-8f543df84ecc.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190065-a64aa180-c8a5-11eb-9331-a772912478b1.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190067-a6e33800-c8a5-11eb-8ab2-2617efb3f44d.png" width="40%" height="40%"></img>

- AttGAN : PSNR 30.26 / SSIM 0.9482
- Ours : PSNR 35.92 / SSIM 0.9813    
<img src="https://user-images.githubusercontent.com/74402562/121190071-a6e33800-c8a5-11eb-8b9f-47c06257f5db.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121189972-916e0e00-c8a5-11eb-85b5-7a90d43b6960.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190073-a77bce80-c8a5-11eb-9eb6-7f704afb6db2.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190076-a77bce80-c8a5-11eb-878d-8da4e8f741dc.png" width="40%" height="40%"></img>

- AttGAN : PSNR 26.86 / SSIM 0.9576
- Ours : PSNR 26.90 / SSIM 0.9756    
<img src="https://user-images.githubusercontent.com/74402562/121190078-a8146500-c8a5-11eb-9327-c005bc4b7e29.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121189976-929f3b00-c8a5-11eb-84b3-32d930b57381.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190080-a8146500-c8a5-11eb-9530-f166b56c1abf.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190108-afd40980-c8a5-11eb-8f57-6c1d157f1349.png" width="40%" height="40%"></img>

- AttGAN : PSNR 24.33 / SSIM 0.9482
- Ours : PSNR 36.05 / SSIM 0.9835    
<img src="https://user-images.githubusercontent.com/74402562/121190113-b06ca000-c8a5-11eb-8587-23f47744bcbf.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121189984-93d06800-c8a5-11eb-8ef2-464ab3847163.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190115-b1053680-c8a5-11eb-900b-dca3e1192a44.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190116-b1053680-c8a5-11eb-85db-e05e32937405.png" width="40%" height="40%"></img>

- AttGAN : PSNR 27.13 / SSIM 0.9548
- Ours : PSNR 34.10 / SSIM 0.9792    
<img src="https://user-images.githubusercontent.com/74402562/121190119-b19dcd00-c8a5-11eb-95a8-af4274443235.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190121-b19dcd00-c8a5-11eb-98d7-98a8d073eee8.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190121-b19dcd00-c8a5-11eb-98d7-98a8d073eee8.png" width="40%" height="40%"></img>
<img src="https://user-images.githubusercontent.com/74402562/121190123-b2366380-c8a5-11eb-9563-7e71217190d9.png" width="40%" height="40%"></img>

----------------------------
### Result of Object Detection YOLO v3
- Raindrop Image & Generative Image   

![rain_124](https://user-images.githubusercontent.com/74402562/119976996-d450fb80-bff2-11eb-838f-d02a475c591c.png)
![result_124](https://user-images.githubusercontent.com/74402562/119976994-d450fb80-bff2-11eb-9660-7c39875eeef2.png)

![input_447](https://user-images.githubusercontent.com/74402562/119976548-32311380-bff2-11eb-9069-66ea85fe7d39.png)
![result_447](https://user-images.githubusercontent.com/74402562/119976544-30ffe680-bff2-11eb-968d-a3c84e713bc3.png)

![input_683](https://user-images.githubusercontent.com/74402562/119976988-d2873800-bff2-11eb-944d-7b4f59928865.png)
![result_683](https://user-images.githubusercontent.com/74402562/119976986-d1eea180-bff2-11eb-8b94-c488d91c72cc.png)

![input_788](https://user-images.githubusercontent.com/74402562/119976990-d31fce80-bff2-11eb-9345-9805ddea2597.png)
![result_788](https://user-images.githubusercontent.com/74402562/119976989-d31fce80-bff2-11eb-821c-df9d12dd5dec.png)

![input_215](https://user-images.githubusercontent.com/74402562/119978713-ef246f80-bff4-11eb-8079-0b89a1b1319e.png)
![result_215](https://user-images.githubusercontent.com/74402562/119978724-f186c980-bff4-11eb-9e48-ec61f71285f7.png)

![input_950](https://user-images.githubusercontent.com/74402562/119976993-d3b86500-bff2-11eb-983c-0020c8e339eb.png)
![result_950](https://user-images.githubusercontent.com/74402562/119976991-d3b86500-bff2-11eb-9432-1ea42bfb8b1e.png)

-----------------------------
### References

- LSGAN : https://arxiv.org/pdf/1611.04076.pdf
- SRGAN : https://arxiv.org/pdf/1609.04802.pdf
- RCAN(Image Super-Resolution Using Very Deep Residual Channel Attention Networks) : https://arxiv.org/pdf/1807.02758.pdf
- pix2pix : https://arxiv.org/pdf/1611.07004.pdf
- Neural style transfer : https://arxiv.org/pdf/1508.06576v2.pdf
- Loss Functions for Image Restoration with Neural Networks : https://arxiv.org/pdf/1511.08861.pdf
