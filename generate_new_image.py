import matplotlib.pyplot as plt
import tensorflow  as tf 

def generate_samples(model, diffusion_process, timesteps=1000):

    noise = tf.random.normal(shape=(1, 32, 32, 3))

    diffused_noise = diffusion_process(noise, model, timesteps=timesteps)


    for i in range(10):

        logits = model(diffused_noise, training=False)

        probs = tf.nn.softmax(logits)

        sample = tf.random.categorical(probs, num_samples=1)
        a = tf.random.categorical(probs, )
        sample = tf.squeeze(sample, axis=-1)

        sample = tf.cast(sample, tf.float32) / 255.0

        diffused_noise = diffusion_process(sample, model, timesteps )

        plt.imshow(sample.numpy())

        plt.show()