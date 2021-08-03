# Document-Classification-Using-Images
Our paper at:  
https://doi.org/10.1093/bioinformatics/btab331  

## Codes

**Step 1: Generate image based document representation**  

**Step 1.1:** Build image classifier  
Command: python code/Step1_1_Data_preparation_image_classification.py  
Inputs:  
*Img_data_path:* The annotated image dataset used for image classification.  
Outputs:  
The image classifier (in h5 format) for assigning class labels to extracted panels.  

**Step 1.2:** Create Figure-words  
Command: python code/Step1_2_creating_type_pattern_feature.py  
Inputs:  
*Clf_results_path:* Image classification results.  
*Data_path:* The path to the json file containing basic information of each document, such as title-and-abstract, caption list, and figure list.  
Outputs:  
The updated json file contains all basic document information including Figure-words information.  

**Step 1.3:** Generate Figure-word based document representation  
Command: python code/Step1_3_BDC_using_image_type_pattern.py  
Inputs:  
*Data\_path:* The path to the json file passed from Step 1.2.  
Outputs:  
The updated json file containing image-based document representation for each document.  

**Step 2: Caption based document representation**  
Command: python code/Step2_BDC_using_captions.py  
Inputs:  
*Data\_path:* The path to the json file contains all basic document information.  
Outputs:  
The updated json file containing the caption-based document representation for each document.  

**Step 3:** Title-and-abstract based document representation  
Command: python code/Step2_BDC_using_TitleAbstract.py  
Inputs:  
*Data\_path:* The path to the json file contains all basic document information.  
Outputs:  
The updated json file containing the title-and-abstract based document representation for each document.  

**Step 4:** Information integration for document classification  
**Step 4.1:** Integration via concatenated vectors  
Command: python code/Step4_1_BDC_using_concatenated_vectors.py  
Inputs:  
*Data\_path:* The path to the json file containing the image-based, caption-based and title-and-abstract based document representations for each document.  
Outputs:  
Class labels assigned to documents.  

**Step 4.2:** Integration via meta-classification  
Command: python code/Step4_2_BCD_using_meta_classification.py  
Inputs:  
Data_path: The path to the json file containing the class label and probability assigned by base classifiers for each document.  
Outputs:  
Class labels assigned to documents.  

## Datasets  

**GXD2000 dataset**  
Dataset path: /datasets/GXD2000  
Description: This dataset is a subset of the dataset used by [Jiang et al. (2017)](https://doi.org/10.1093/database/bax017) . The original dataset contains PDF files of 58,362 publications curated by the Jackson Lab’s Gene Expression Database. We selected at random 1,000 relevant and 1,000 irrelevant documents from these publications.  

**DSP dataset**  

Dataset path: /datasets/DSP  

Description: This dataset was introduced by [Burns et al. (2019)](https://doi.org/10.1093/database/baz034). The original dataset comprises 537 publications relevant to molecular interactions and 451 irrelevant ones. In our dataset, 534 relevant publications and 448 irrelevant ones are retained as their PDF files are available for download online.  

