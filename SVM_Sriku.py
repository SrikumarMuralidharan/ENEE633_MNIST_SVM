import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import argparse

def SVM_func(tr_img, te_img, tr_lbl, te_lbl, te_img1,trans):
    k = input("Choice of Kernel: type integers: rbf-1, poly-2, linear-3: \n ")
    if k == '1':
        print('rbf kernel chosen \n')
        kernel = 'rbf'
    elif k == '2':
        print('poly kernel chosen \n')
        kernel = 'poly'
    else:
        print('linear kernel chosen \n')
        kernel = 'linear'
    model = SVC(C=1,kernel=kernel)

    # fitting labels and images for training data
    print("Fitting model")
    model.fit(tr_img, tr_lbl)
    
    # Training accuracy and creating confusion matrix:
    pred_tr_lbl = model.predict(tr_img)
    print("\nTraining Accuracy = ", metrics.accuracy_score(y_true=tr_lbl, y_pred=pred_tr_lbl))
    tr_CM = metrics.confusion_matrix(tr_lbl, pred_tr_lbl)
    tr_CM_disp = metrics.ConfusionMatrixDisplay(tr_CM).plot()
    
                
    # Testing accuracy and creating confusion matrix:
    pred_te_lbl = model.predict(te_img)
    print("\nTesting Accuracy = ", metrics.accuracy_score(y_true=te_lbl, y_pred=pred_te_lbl))
    te_CM = metrics.confusion_matrix(te_lbl, pred_te_lbl)
    te_CM_disp = metrics.ConfusionMatrixDisplay(te_CM).plot()
    
    # Plotting predictions:
    fig, axis = plt.subplots(3,3,figsize=(10,10))
    for i, a in enumerate(axis.flat):
        a.imshow(te_img1[i], cmap='binary')
        a.set(title = f'Act - {te_lbl[i]}Pred - {pred_te_lbl[i]}')
    plt.savefig(f'{trans}_{kernel}_results.png')
    plt.close()
            

def Main():
    # Loading testing and training data
    (tr_img, tr_lbl), (te_img, te_lbl) = tf.keras.datasets.mnist.load_data()
    
    # to standardize test and training data by removing mean and scaling to unit variance
    # We use fit and transform on the training data and just transform on testing data
    # becuase we dont want to bias model with info from test data.
    ss = StandardScaler()
        
    # # Showing 6 images first to verify dataset is correct
    # for i in range(6):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(tr_img[i])
    
        
    # Preprocessing training and testing data
    tr_img = tr_img.reshape((60000,784))    # 28*28 = 784 - flattening image for fit_transform
    tr_img = tr_img/255.0                   # Normalising data to a scale of 0 to 1
    tr_img = ss.fit_transform(tr_img)       # Reason explained above
    
    te_img1 = te_img                        # Storing data for future prediction
    te_img = te_img.reshape((10000,784))    # 28*28 = 784 - flattening image for transform
    te_img = te_img/255.0                   # Normalising data to a scale of 0 to 1
    te_img = ss.transform(te_img)           # Reason explained above
    
    choice = input("Choice of transform: PCA - 1, LDA - 2, none - anything: ")
    if choice == '1':
        print("You chose PCA")
        trans = "PCA"
        pca = PCA(n_components=0.97)        # Should lie between 0.95 to 0.99
        tr_img = pca.fit_transform(tr_img)  # 
        te_img = pca.transform(te_img)
        
    elif choice == '2':
        print("You chose LDA")
        trans = "LDA"
        lda = LDA(n_components=9)        # For M=10 Classes, we can have a max of M-1 
        tr_img = lda.fit_transform(tr_img, tr_lbl)
        te_img = lda.transform(te_img)
    
    else:
        trans = "none"
        
    SVM_func(tr_img, te_img, tr_lbl, te_lbl, te_img1,trans)                      
if __name__ == '__main__':
    Main()