
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
import numpy as np 

tf.config.set_visible_devices([],'GPU')

def make_diffusion_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    for i in range(10):
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
    model.add(layers.Conv2D(3, (1, 1), padding='same')) 
    return model

def diffusion_process(x,model, timesteps=1000):
    
    for i in range(timesteps):
        # print(i)
        noise = tf.random.normal(shape=x.shape)

        x = x + tf.math.sqrt(2.0 * 0.01) * noise

        x = x / tf.math.sqrt(1.0 + 2.0 * 0.01)

        x = model(x)

    return x


def train_diffusion_model(model):

    # dataset = tfds.load('cifar10', split='train', shuffle_files=True)
    dataset = tf.keras.datasets.cifar10.load_data()
    train, test = dataset
    x_train, y_train = train
    x_test, y_test = test 
    
    del train, test, dataset,y_train,  y_test
    ## Normalization 0 -1 
    x_train = x_train / 255
    x_test = x_test / 255
    
    x = tf.concat((x_train, x_test), axis = 0)
    del x_train, x_test
    x = np.float32(x.numpy())
    x = x[:200,...]
    # dataset = dataset.batch(64).prefetch(tf.data.AUTOTUNE)
    
    batch_sz  = 128 #64 
    N  =  len(x)
    n_bz = N // batch_sz 
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    for epoch in range(100):
        # x = tf.random.shuffle(x)
        x = shuffle(x)
        for j in range(n_bz):
            print(j)
            batch =  x[j*batch_sz : (j+1)  * batch_sz, ...]
            with tf.GradientTape() as tape:

                noise = tf.random.normal(shape=batch.shape)

                diffused_noise = diffusion_process(noise, model, timesteps = 10)

                logits = model(diffused_noise, training=True)

                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(batch, logits))
                
            gradients = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# Save the weights of the trained model
model = make_diffusion_model()
train_diffusion_model(model)
model.save_weights('diffusion_model.h5')



# # Load the saved weights into a new model object

new_model = make_diffusion_model()

new_model.load_weights('diffusion_model.h5')

from  generate_new_image import generate_samples
generate_samples(model, diffusion_process,  timesteps=10)