## Fingerprint and Finger-vein fusion recognition

The code for the paper **Robust Graph Fusion and Recognition Framework for Fingerprint and Finger-vein**

#### Requirements

- python 3.7
- pytorch 1.9.1
- torch-geometric 2.0.1
- scipy 1.7.1

#### Testing a Model

We provide a pretrained model `"model/model_dict.pth"` and a small set of test samples `data/FP_200sp_test_c2.00.pkl` and `data\FV_200sp_test_c2.00.pkl`

Run a test script `python3 main.py --train False`

#### Training a Model

Run a test script `python3 main.py --train True`

#### Data samples

<img src="data/fp0000.bmp" alt="fp0000" style="zoom:80%;" /><img src="C:\Users\QU\Desktop\IET\project\CGCN\data\fp0040.bmp" alt="fp0040" style="zoom:80%;" /><img src="C:\Users\QU\Desktop\IET\project\CGCN\data\fp0080.bmp" alt="fp0080" style="zoom:80%;" /><img src="C:\Users\QU\Desktop\IET\project\CGCN\data\fp0130.bmp" alt="fp0130" style="zoom:80%;" />



<img src="C:\Users\QU\Desktop\IET\project\CGCN\data\fv0000.bmp" alt="fv0000" style="zoom:100%;" />       <img src="C:\Users\QU\Desktop\IET\project\CGCN\data\fv0010.bmp" alt="fv0010" style="zoom:100%;" />       <img src="C:\Users\QU\Desktop\IET\project\CGCN\data\fv0040.bmp" alt="fv0040" style="zoom:100%;" />        <img src="C:\Users\QU\Desktop\IET\project\CGCN\data\fv0140.bmp" alt="fv0140" style="zoom:100%;" />













