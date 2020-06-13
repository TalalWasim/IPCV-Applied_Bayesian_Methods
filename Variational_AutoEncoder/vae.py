# Implements auto-encoding variational Bayes.
import os
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import flatten  # This is used to flatten the params (transforms a list into a numpy array)


# images is an array with one row per image, file_name is the png file on which to save the images
def save_images(images, file_name): return s_images(images, file_name, vmin=0.0, vmax=1.0)


# Sigmoid activation function to estimate probabilities
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Relu activation function for non-linearity
def relu(x):
    return np.maximum(0, x)


# This function initializes the parameters of a deep neural network
def init_net_params(layer_sizes, scale=1e-2):
    """Build a (weights, biases) tuples for all layers."""

    return [(scale * npr.randn(m, n),  # weight matrix
             scale * npr.randn(n))  # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


# This will be used to normalize the activations of the NN
# This computes the output of a deep neural network with params a list with pairs of weights and biases
def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)  # nonlinear transformation

    # Last layer is linear
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs


# This implements the reparametrization trick
def sample_latent_variables_from_posterior(encoder_output):
    # Params of a diagonal Gaussian.
    D = np.shape(encoder_output)[-1] // 2
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    # We use the reparametrization trick to generate one sample from q(z|x) per each batch datapoint
    # The output of this function is a matrix of size the batch x the number of latent dimensions

    batch_size, latent_dim = mean.shape[0], mean.shape[1]
    samples = npr.randn(batch_size, latent_dim)

    return mean + samples * np.exp(log_std)


# This evaluates the log of the term that depends on the data
def bernoulli_log_prob(targets, logits):
    # logits are in R
    # Targets must be between 0 and 1

    # Compute the log probability of the targets given the generator output specified in logits
    # We sum the probabilities across the dimensions of each image in the batch.
    # The output of this function is a vector of size the batch size

    # converting the generator output to probabilities too
    logits = sigmoid(logits)
    # implementing the log of equation 3
    output = np.sum(np.log(targets * logits + (1 - targets) * (1 - logits)), axis=1)

    return output


# This evaluates the KL between q and the prior
def compute_KL(q_means_and_log_stds):
    D = np.shape(q_means_and_log_stds)[-1] // 2
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]

    # Compute the KL divergence between q(z|x) and the prior (we use a standard Gaussian for the prior)
    # We use the fact that the KL divervence is the sum of KL divergence of the marginals if q and p factorize
    # The output of this function is a vector of size the batch size

    # implement equation 12
    std = np.exp(log_std)
    KL = np.sum(0.5 * (np.square(std) + (np.square(mean) - 1 - 2*log_std)), axis=1)

    return KL


# This evaluates the lower bound
def vae_lower_bound(gen_params, rec_params, data):
    # Compute a noisy estimate of the lower bound by using a single Monte Carlo sample:

    # 1 - We compute the encoder output using neural_net_predict given the data and rec_params
    encoder_output = neural_net_predict(rec_params, data)

    # 2 - We sample the latent variables associated to the batch in data 
    #   (We use sample_latent_variables_from_posterior and the encoder output)
    latent_variable_samples = sample_latent_variables_from_posterior(encoder_output)

    # 3 - We use the sampled latent variables to reconstruct the image and to compute the log_prob of the actual data
    #   (We use neural_net_predict for that)
    generator_output = neural_net_predict(gen_params, latent_variable_samples)
    log_probs = bernoulli_log_prob(data, generator_output)

    # 4 - We compute the KL divergence between q(z|x) and the prior
    #   (We use compute_KL for that)
    KL = compute_KL(encoder_output)

    # 5 - We return an average estimate (per batch point) of the lower bound
    #   by substracting the KL to the data dependent term
    average_estimate = log_probs - KL

    return np.mean(average_estimate)


if __name__ == '__main__':

    # Model hyper-parameters

    npr.seed(0)  # We fix the random seed for reproducibility

    latent_dim = 50
    data_dim = 784  # How many pixels in each image (28x28).
    n_units = 200
    n_layers = 2

    gen_layer_sizes = [latent_dim] + [n_units for i in range(n_layers)] + [data_dim]
    rec_layer_sizes = [data_dim] + [n_units for i in range(n_layers)] + [latent_dim * 2]

    # Training parameters

    batch_size = 200
    num_epochs = 30
    learning_rate = 0.001

    print("Loading training data...")
    N, train_images, _, test_images, _ = load_mnist()

    # Parameters for the generator network p(x|z)
    init_gen_params = init_net_params(gen_layer_sizes)

    # Parameters for the recognition network p(z|x)
    init_rec_params = init_net_params(rec_layer_sizes)
    combined_params_init = (init_gen_params, init_rec_params)
    num_batches = int(np.ceil(len(train_images) / batch_size))

    # We flatten the parameters (transform the lists or tupples into numpy arrays)
    flattened_combined_params_init, unflat_params = flatten(combined_params_init)

    # Actual objective to optimize that receives flattened params
    def objective(flattened_combined_params):

        combined_params = unflat_params(flattened_combined_params)
        data_idx = batch
        gen_params, rec_params = combined_params

        # We binarize the data
        on = train_images[data_idx, :] > npr.uniform(size=train_images[data_idx, :].shape)
        images = train_images[data_idx, :] * 0.0
        images[on] = 1.0

        return vae_lower_bound(gen_params, rec_params, images)

    # Get gradients of objective using autograd.
    objective_grad = grad(objective)
    flattened_current_params = flattened_combined_params_init

    # ADAM parameters
    t = 1

    ####---- Task 2: Defining the ADAM parameters ---####

    alpha = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    # We initialize m and v to be the size of the flattened_current_params
    m = np.zeros_like(flattened_current_params)
    v = np.zeros_like(flattened_current_params)

    # We do the actual training
    for epoch in range(num_epochs):    
        elbo_est = 0.0

        for n_batch in range(int(np.ceil(N / batch_size))):
            batch = np.arange(batch_size * n_batch, np.minimum(N, batch_size * (n_batch + 1)))
            grad = objective_grad(flattened_current_params)

            ####---- Task 2: Calculating ADAM paramers and updating the model params ---####

            # Calculate the parameters of the ADAM optimizer
            m = beta_1*m + (1-beta_1)*grad
            v = beta_2*v + (1-beta_2)*(grad*grad)
            m_hat = m/(1-(beta_1**t))
            v_hat = v/(1-(beta_2**t))

            # Update the parameters using ADAM optimizer
            flattened_current_params = flattened_current_params + (alpha*m_hat)/(np.sqrt(v_hat) + epsilon)
            elbo_est += objective(flattened_current_params)
            
            t += 1

        print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N / batch_size)))

    # We save the trained params so we don't have to retrain each time
    np.save(os.path.join("trained_params", "data.npy"), flattened_current_params)

    # We obtain the final trained parameters
    flattened_current_params = np.load(os.path.join("trained_params", "data.npy"))

    gen_params, rec_params = unflat_params(flattened_current_params)

    ####---- Task 3.1 ---####
    
    # We generate 25 samples from the prior.
    # Note the prior P(z) is a standard Gaussian
    num_prior_images = 25
    z = npr.randn(num_prior_images, latent_dim)

    # Generate the images using the prior and gen params
    generated_images = neural_net_predict(gen_params, z)

    # Convert the logits to probabilities
    sigmoid_generated_images = sigmoid(generated_images)
    save_images(sigmoid_generated_images, os.path.join("saved_images", "gen_prior_25.png"))

    ####---- Task 3.2 ---####
    
    # Select specific number of test images
    num_test_images = 10
    test_images = test_images[0:num_test_images, :]

    # Generate encoded output for the test images
    test_encoder_output = neural_net_predict(rec_params, test_images)

    # Sample latent variables from the encoded representation
    test_latent_variable_samples = sample_latent_variables_from_posterior(test_encoder_output)

    # Generate new images using the sampled latent representation and apply sigmoid
    test_generated_images = neural_net_predict(gen_params, test_latent_variable_samples)
    test_sigmoid_generated_images = sigmoid(test_generated_images)

    # Concatanate the original images and save to file
    test_concat = np.concatenate((test_images, test_sigmoid_generated_images))
    save_images(test_concat, os.path.join("saved_images", "gen_test_10.png"))

    ####---- Task 3.3 ---####

    num_interpolations = 5
    for i in range(5):
        # Select pairs of consecutive images
        I = test_images[i * 2, :]
        G = test_images[i * 2 + 1, :]

        # Generate the encoded output of the images
        I_encoder_output = neural_net_predict(rec_params, I)
        G_encoder_output = neural_net_predict(rec_params, G)

        # Use the mean as latent representation
        D = np.shape(I_encoder_output)[-1] // 2
        I_mean = I_encoder_output[:D]
        G_mean = G_encoder_output[:D]

        # Generating 25 numbers between 0 and 1 for interpolation
        s = np.linspace(0, 1, num=25)

        # Create convex combinations of the two images
        latent_rep = np.outer(s, G_mean) + np.outer((1 - s), I_mean)

        # Generate output images for the convex combinations and apply sigmoid
        gen_images = neural_net_predict(gen_params, latent_rep)
        sig_gen_images = sigmoid(gen_images)

        # Save each set of interpolated images to separate file
        save_images(sig_gen_images, os.path.join("saved_images", "gen_interpolated_{}.png".format(i + 1)))
