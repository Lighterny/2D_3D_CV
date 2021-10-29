# Welcome to 2D_3D_CV!
### Firstly, what is this about? 
This is a small computer vision study comparing 2D vs 3D CNN architectures in classifying lung disease (TB) from chest CT scans. More specifically, I compare three CNN models of increasing complexity (baseline, + batch normalisation, ++ more layers) in two parallel experiments, 2D and 3D equivalents. In the end, I compared 6 in-house designed CNN's, and out of curiosity also included a RESNET50 transfer learning model.

### The contents include: 
- **Python Scripts & Notebook**: Python scripts & notebooks!
  - [**2D Models**](https://github.com/Lighterny/2D_3D_CV/tree/main/Python%20Scripts%20%26%20Notebook/2D%20Models): python scripts showing 3x 2D CNN designs.
  - [**3D Models**](https://github.com/Lighterny/2D_3D_CV/tree/main/Python%20Scripts%20%26%20Notebook/3D%20Models): python scripts of 3x 3D CNN models - these are the 3D equivalents of the above 2D models.
  - [**MosMedCT_EDA.ipynb**](https://github.com/Lighterny/2D_3D_CV/blob/main/Python%20Scripts%20%26%20Notebook/MosMedCT_EDA.ipynb): Exploratory analysis and pre-processing of this CT dataset.
  - [**Run_Models.ipynb**](https://github.com/Lighterny/2D_3D_CV/blob/main/Python%20Scripts%20%26%20Notebook/Run_Models.ipynb): This is where I use Sagemaker notebook to deploy all these models in an EC2 instance in AWS.
- **README.md**: Hello, you're reading me.
- [**Report.pdf**](https://github.com/Lighterny/2D_3D_CV/blob/main/Report.pdf): An academic-style report explaining the motivation of the study and reporting the results. Spoiler alert, 3D wins, but the final results may still surprise you, so I encourage you to have a look at the pdf report. 
