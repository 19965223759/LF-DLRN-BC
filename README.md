# Deep learning radiomics explores new biomarkers affecting breast cancer distant metastasis from CT imaging

## 1. Model schematic diagram.

![](https://github.com/19965223759/LF-DLRN-BC/blob/main/schematic%20diagram.png)

The images on T4 and T11 and CPBC were respectively put into DenseNet161, ResNet50 and GradientBoosting for classification, and late_fusion_prob was obtained by late fusion method. The significant clinicopathological features obtained by multivariate Cox regression and late_fusion_prob were used as predictors to construct the nomogram. Finally, the KM curve was adopted to analyze the survival of patients with distant metastases.

## 2. Requirements

- Python3.9.13
- Pytorch1.12.1 + cuda11.6
- R (V 4.1.2)

## 3. Usage

### 3.1 dateset

CT images and clinicopathological information of the fourth and eleventh thoracic vertebrae in 431 patients with breast cancer. 

### 3.2 train the DenseNet161 model

You need to train the model with the following commands:

```Py
$ python T4_probability.py
```

This file trains DenseNet161 on images of the T4 horizontal thoracic vertebrae so that it can estimate the probability and result that each image in the test cohort will be correct.

### 3.3 train the ResNet50 model

You need to train the model with the following commands:

```py
$ python T11_probability.py
```

This file trains ResNet50 on images of the T11 horizontal thoracic vertebrae so that it can estimate the probability and result that each image in the test cohort will be correct.

###  3.4 train the GradientBoosting model

You need to train the model with the following commands:

```py
$ python GradientBoostingClassifier.py
```

This file trains the CPBC data  to obtain test cohort prediction results.

### 3.5 late fusion 

The images on T4 and T11 and CPBC were respectively put into the selected DenseNet161, ResNet50 and GradientBoosting to predict BC distant metastasis. The prediction results of the three models were fused by the majority voting strategy. If the prediction result of the images on T4 was consistent with that the images on T11, the prediction probability of the T4 remained unchanged; if it was not, the fusion results were used for judgment. If the prediction result of images on T4 was consistent with the fusion result, the prediction probability of T4 would be continued. If not, the prediction probability of images on T11 would be used to replace the corresponding position probability of images on T4, so as to obtain the late fusion probability (late_fusion_prob).

### 3.6 Nomogram + calibration + DCA

You need to execute the following files in R software:

- R_nom.R
- calibration.R
- dca.R
- KM_OS.R
- KM_DMFS.R

After executing the R file, you will get a nomogram-3, a calibration-3, a DCA curve and two KM curves.

## 4. Results

The experimental results are in LF-DLRN and R4.1.2 folders.



