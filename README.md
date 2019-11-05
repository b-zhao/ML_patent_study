# ML_patent_study

To prepare input data (X) and labels (Y) for the deep learning model, first copy Database_Patents_MLClass_Sample_Sep2019.dta or .csv to the data_preparation folder, then run preprocessing.py.


To do: 
1. Change framework to PyTorch so that we can use GPU
2. Using classes and subclasses of patents for classification.
3. Using Lifecycle variable for predicting delays. This is the time duration that has passed from the first patent in the same technological class. Research shows that the patents applied in the beginning of the technology lifecycle would take longer to get granted as compared to the patents applied later in the technology lifecycle (Ref. Regibeau & Rockett (2010)).  
4. Try different models (neural network), or tune parameters for the current SVM
5. Add an unsupervised method
