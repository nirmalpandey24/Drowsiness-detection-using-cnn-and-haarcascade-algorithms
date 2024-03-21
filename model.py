import cv2
import os
import numpy as np

class Conv2D:
    def __init__(self, num_filters, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size) / (kernel_size * kernel_size)

    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h - self.kernel_size + 1):
            for j in range(w - self.kernel_size + 1):
                region = image[i:(i + self.kernel_size), j:(j + self.kernel_size)]
                yield region, i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.kernel_size + 1, w - self.kernel_size + 1, self.num_filters))

        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))

        return output

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for image_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * image_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # Return gradients for next layer
        return None

class MaxPool2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
    
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // self.pool_size
        new_w = w // self.pool_size

        for i in range(new_h):
            for j in range(new_w):
                region = image[(i * self.pool_size):(i * self.pool_size + self.pool_size),
                               (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                yield region, i, j
    
    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))

        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_input = np.zeros(self.last_input.shape)

        for region, i, j in self.iterate_regions(self.last_input):
            h, w, f = region.shape
            amax = np.amax(region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * self.pool_size + i2, j * self.pool_size + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input


class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input_flat = input.flatten()
        self.last_input = input_flat
        input_len, nodes = self.weights.shape

        totals = np.dot(input_flat, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backward(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            t_exp = np.exp(self.last_totals)

            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)

class SimpleCNN:
    def __init__(self, input_shape, num_classes):
        self.conv = Conv2D(8, 3)
        self.pool = MaxPool2D(2)
        self.softmax = Softmax((input_shape[0] // 2 - 1) * (input_shape[1] // 2 - 1) * 8, num_classes)
    
    def forward_pass(self, input):
        conv_out = self.conv.forward(input)
        pool_out = self.pool.forward(conv_out)
        softmax_out = self.softmax.forward(pool_out)
        return softmax_out
    
    def train(self, X_train, y_train, epochs=10, learn_rate=0.01):
        for epoch in range(epochs):
            print("Epoch %d" % (epoch + 1))
            for i, image in enumerate(X_train):
                probs = self.forward_pass(image)
                loss = -np.log(probs[y_train[i]])
                acc = 1 if np.argmax(probs) == y_train[i] else 0
                print("Step %d - Loss: %.4f - Accuracy: %d" % (i+1, loss, acc))
                # Backpropagation
                d_L_d_out = np.zeros(num_classes)
                d_L_d_out[y_train[i]] = -1 / probs[y_train[i]]
                d_L_d_inputs = self.softmax.backward(d_L_d_out, learn_rate)
                d_L_d_inputs = self.pool.backward(d_L_d_inputs, learn_rate)
                d_L_d_inputs = self.conv.backward(d_L_d_inputs, learn_rate)

# Load and preprocess data
def load_data(folder_path):
    images = []
    labels = []
    label_map = {'open': 0, 'closed': 1}  # Map folder names to integer labels
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                if file.endswith('.png'):
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    image = cv2.resize(image, (100, 100))  # Resize images to 100x100
                    image = image / 255.0  # Normalize pixel values
                    images.append(image)
                    labels.append(label_map[label])  # Map folder name to integer label
    return np.array(images), np.array(labels)

# Define paths to train and test folders
train_folder = r'F:\Drowsiness detection\dataset\train'
test_folder = r'F:\Drowsiness detection\dataset\test'

# Load train and test data
X_train, y_train = load_data(train_folder)
X_test, y_test = load_data(test_folder)

# Define input shape and number of classes
input_shape = (100, 100)
num_classes = 2  # Two classes: open eye and closed eye

# Initialize and train the CNN model
cnn_model = SimpleCNN(input_shape, num_classes)
cnn_model.train(X_train, y_train)
