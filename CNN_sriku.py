import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, normalization
from keras.utils.vis_utils import plot_model
from sklearn import metrics
 
    
def CNN_model():    
   
    # Defining our CNN Model
    model = Sequential()
    # First convolutional layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
    # Add a maxpool layer
    model.add(MaxPool2D(pool_size=(2,2)))               # Default pool size
    # Applying Batch normalization
    model.add(normalization.BatchNormalization())       #Keeps mean activation ~0, SD ~0
    # Second convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    # Add a maxpool layer
    model.add(MaxPool2D(pool_size=(2,2)))               # Default pool size
    # Applying Batch normalization
    model.add(normalization.BatchNormalization())       #Keeps mean activation ~0, SD ~0
    # Fourth convolutional layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    # Add a maxpool layer
    model.add(MaxPool2D(pool_size=(2,2)))               # Default pool size
    # Covert from 2D to 1D
    model.add(Flatten())
    # Applying Batch normalization
    model.add(normalization.BatchNormalization())       #Keeps mean activation ~0, SD ~0
    # Adding a Dense layer
    model.add(Dense(256,activation="relu"))    
	# Output layer
    model.add(Dense(10,activation="softmax"))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

def Main():
    
    epochs = 50
    batch_size = 64
    
    # Loading testing and training data
    (tr_img, tr_lbl), (te_img, te_lbl) = tf.keras.datasets.mnist.load_data()
    # Preprocessing training and testing data
    tr_img = tr_img.reshape(60000,28,28,1)      # 28*28 = 784 - flattening image for fit_transform
    tr_img = tr_img/255.0                       # Normalising data to a scale of 0 to 1
    te_img1 = te_img                            # Storing data for future prediction
    te_img = te_img.reshape(10000,28,28,1)      # 28*28 = 784 - flattening image for transform
    te_img = te_img/255.0                       # Normalising data to a scale of 0 to 1
    
    # Categorising testing and training labels set based on 10 classes (0-9)
    tr_lbl  = to_categorical(tr_lbl,10)
    te_lbl  = to_categorical(te_lbl,10)
    
    # Steps to prevent overfitting
    datagen = ImageDataGenerator(
        featurewise_center=False,               # input zero mean
        samplewise_center=False,                # samplewise zero mean
        featurewise_std_normalization=False,    # input/std
        samplewise_std_normalization=False,     # samplewise input/std 
        zca_whitening=False,                    # dont apply ZCA whitening
        rotation_range=45,                      # range 0 - 180
        zoom_range = 0.2,                       # zoom 
        width_shift_range=0.1,                  # img shift width-wise
        height_shift_range=0.1,                 # img shift height-wise
        horizontal_flip=False,                  # randomly flip images
        vertical_flip=False)                    # randomly flip images
    
    # Using datagen to fit training dataset
    datagen.fit(tr_img)
    
    # attaching labels to changed files:
    gen_tr = datagen.flow(tr_img, tr_lbl, batch_size=64)
    gen_te = datagen.flow(te_img, te_lbl, batch_size=64)
    
    model = CNN_model()
    
    # model = load_model(f'model_{epochs}_epochs_{batch_size}_batch.h5')
    
    history = model.fit_generator(gen_tr, 
                              epochs = epochs, 
                              steps_per_epoch = tr_img.shape[0] // batch_size,
                              validation_data = gen_te,
                              validation_steps = te_img.shape[0] // batch_size)
    
    model.save(f'model_{epochs}_epochs_{batch_size}_batch.h5')

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    
    # Evaluating Test and train accuracies for this model
    _, test_accuracy = model.evaluate_generator(gen_te, steps=te_img.shape[0]//batch_size, verbose=0)
    print("Test accuracy = "+str(test_accuracy*100))
    _, train_accuracy = model.evaluate_generator(gen_tr, steps=tr_img.shape[0]//batch_size, verbose=0)
    print("Train accuracy = "+str(train_accuracy*100))
    
    # Plots for training vs Validation accuracy:
    plt.figure(figsize=(10,10))
    plt.title('Training vs Validation accuracies')
    plt.plot(range(epochs),history.history['accuracy'], label='Training accuracy')
    plt.plot(range(epochs),history.history['val_accuracy'], label='Validation accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.close()
    
    # Plots for training vs Validation loss:
    plt.figure(figsize=(10,10))
    plt.title('Training vs Validation losses')
    plt.plot(range(epochs),history.history['loss'], label='Training loss')
    plt.plot(range(epochs),history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()
    
    
    
    pred_te_lbl1 = model.predict(te_img)
    pred_te_lbl = np.argmax(pred_te_lbl1, 1)
    te_lbl1 = np.argmax(te_lbl, 1)
    
    # Confusion matrix for testing data
    te_CM = metrics.confusion_matrix(te_lbl1, pred_te_lbl)
    te_CM_disp = metrics.ConfusionMatrixDisplay(te_CM).plot()
    
    # Plotting predictions:
    fig, axis = plt.subplots(3,3,figsize=(10,10))
    for i, a in enumerate(axis.flat):
        a.imshow(te_img1[i], cmap='binary')
        a.set(title = f'Act - {te_lbl1[i]} Pred - {pred_te_lbl[i]}')
    plt.savefig('results.png')
    plt.close()
    
    

if __name__ == '__main__':
    Main()