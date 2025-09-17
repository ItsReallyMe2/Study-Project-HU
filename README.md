# **Study Project at Humboldt University 🎓**
This is a collection of some of the code and analyses I carried out during my four-month mandatory internship as part of my biology degree at Humboldt University in Berlin. I completed my internship at the Institute for Theoretical Biology (HU) under the supervision of [Dr. Bharath Ananthasubramaniam](https://github.com/bharathananth). 

***My project focused on exploring dimensionality reduction in RNA sequencing time series data and then progressed to the application of unsupervised machine learning for phase reconstruction of circadian rhythms.***

## The Journey 🛣️:
So it started with understanding the concept of dimensionality reduction, and one of the most well-known tools for this is [PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ&t=28s). At this point, I started looking into the [scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html) library in Python and practiced applying PCA to toy datasets from it. Here is the example:

- [PCA of 3 toy datasets from skikit-learn](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/0_toy_datasets.ipynb)

During this time, I not only established the environment for future programming, but also created a series of [custom visualization functions](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/custom_func.py) that I frequently used throughout my project and bachelor's thesis.

I then proceeded to apply this knowledge to real world data from the [meta-analysis of diurnal transcriptomics in mouse liver by Thomas Brooks](https://zenodo.org/records/7760579)(The data is available via the link.). The analysis encompassed the following steps:

- [Wrangle the data and save it as a convenient AnnData object](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/1_anndata_mouse.ipynb)

I then performed basic preprocessing procedures such as normalization, logarithmic transformation, and removal of excessive features, and later removed the batch effect and saved the changes back to the existing [AnnData](https://anndata.readthedocs.io/en/stable/) object.

- [Preprocessing of the mouse liver dataset](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/2_preprocess_mouse.ipynb)

After that, I began examining PCA and color-coded samples not only by time, but also by other characteristics that researchers provide with dataset in order to find the explanation for PCA results.

- [PCA results of the mouse liver dataset](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/3_pca_mouse.ipynb)

Once I realized that the results could be explained by technical differences or the unique experimental design of each individual study, I attempted to remove this batch effect in the hope that the PCA visualization would reveal a clear circadian signal in the form of a circle/ellipse/donut, that looks like [that](https://www.pnas.org/cms/10.1073/pnas.1619320114/asset/e3b4419f-f0b8-4de3-af90-75dcd149623d/assets/graphic/pnas.1619320114sfig01.jpeg) (from CYCLOPS paper by Ron Anafi) 

- [Batch-Effect removal](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/4_batcheffect_mouse.ipynb)

Having achieved the desired result, I became curious and tried out various other dimensionality reduction tools.

- [Other dimensionality reduction tools](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/5_other_dec_tools.ipynb)

Then I got introduced to [COFE](https://github.com/bharathananth/COFE) and worked with this tool for some time, mainly to see how well it works for phase reconstruction with real data sets of different sizes and from different organisms.

- [Application of COFE to mouse liver dataset](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/6_cofe.ipynb)
- [Application of COFE to 9 different datasets](https://github.com/ItsReallyMe2/Study-Project-HU/tree/main/8_COFE_application)

At the end of my project, I also learn about [CYCLOPS](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/cyclops.py), another tool for phase reconstruction.

- [Application of CYCLOPS on synthetic data and mouse liver data](https://github.com/ItsReallyMe2/Study-Project-HU/blob/main/7_cyclops_test_run.ipynb)

And that's when I came up with the idea for an alternative tool for phase reconstruction, which formed the basis of my bachelor's thesis, which I may elaborate on further later🙈! But for know that it! In summary, the study project not only gave me a solid foundation in programming, but also helped me develop a systematic approach to data analysis and prepared me for more advanced data science techniques, which I am now applying in my bachelor's thesis. 👋🏻

P.S. If you are curious and would like to run the code yourself and need all the data sets, please feel free to contact me!😉
