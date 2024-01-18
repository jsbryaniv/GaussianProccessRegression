
# Import libraries
import numpy as np 
import matplotlib.pyplot as plt


# Define the model class
class GaussianProcess:
    def __init__(self, X, y, sigma, a=None, l=None, eps=None, n_indu=1000):
        """Initializes the Gaussian Process model.
        Inputs:
        X - The data point locations
        y - The data point values
        sigma - The noise standard deviation
        a - The amplitude of the covariance function
        l - The length scale of the covariance function
        n_indu - The number of inducing points
        eps - The size of the regulator
        """

        # Ensure X and y are Nx1 arrays
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)

        # If a and l are not specified, set them from the data
        l = .1*(max(X) - min(X)) if l is None else l  # Length scale
        a = sigma if a is None else a                 # Amplitude of covariance
        eps = .01*sigma**2 if eps is None else eps    # Regulator
        n_data = X.shape[0]                           # Number of data points
        X_indu = np.linspace(min(X), max(X), n_indu)  # Inducing point grid

        # Set Attributes
        self.X = X
        self.y = y
        self.sigma = sigma
        self.a = a
        self.l = l
        self.eps = eps
        self.n_indu = n_indu
        self.n_data = n_data
        self.X_indu = X_indu

    def fit(self):

        # Get values
        a = self.a
        l = self.l
        eps = self.eps
        sigma = self.sigma
        y = self.y
        X = self.X
        X_indu = self.X_indu
        n_indu = self.n_indu
        n_data = self.n_data

        # Create the covariance matrices
        K_data_indu = self.kernel(X, X_indu.T)
        K_indu_indu = self.kernel(X_indu, X_indu.T)
        K_indu_indu_dx = self.kernel(X_indu, X_indu.T, d_dx=1)
        K_indu_indu_dx2 = self.kernel(X_indu, X_indu.T, d_dx=2)
        K_indu_indu_inv = np.linalg.inv(K_indu_indu + eps*np.eye(n_indu))
        K_tilde = np.linalg.inv(
            K_indu_indu_inv + 1/sigma**2 * K_indu_indu_inv @ (K_data_indu.T @ K_data_indu) @ K_indu_indu_inv
        )

        # Get mean curves
        y_indu = K_tilde @ K_indu_indu_inv @ K_data_indu.T @ y / sigma**2
        y_indu_dx = K_indu_indu_dx @ K_indu_indu_inv @ y_indu
        y_indu_dx2 = K_indu_indu_dx2 @ K_indu_indu_inv @ y_indu

        # Store results
        self.K_data_indu = K_data_indu
        self.K_indu_indu = K_indu_indu
        self.K_indu_indu_dx = K_indu_indu_dx
        self.K_indu_indu_dx2 = K_indu_indu_dx2
        self.K_indu_indu_inv = K_indu_indu_inv
        self.K_tilde = K_tilde
        self.y_indu = y_indu
        self.y_indu_dx = y_indu_dx
        self.y_indu_dx2 = y_indu_dx2

        # Return
        return

    def kernel(self, xi, xj, d_dx=0):
        """Returns the kernel matrix for the given data points.
        Input d_dx is the order of the derivative of the kernel with respect to x_i.
        """

        # Get constants
        a = self.a
        l = self.l

        # Get the kernel matrix value
        if d_dx == 0:
            # No derivative
            kij = (
                a**2 * np.exp(-1/(2*l**2) * (xi - xj)**2)
            )
        elif d_dx == 1:
            # First derivative
            kij = (
                a**2 * np.exp(-1/(2*l**2) * (xi - xj)**2) 
                * (-1) * (xi - xj)/l**2
            )
        elif d_dx == 2:
            # Second derivative
            kij = (
                a**2 * np.exp(-1/(2*l**2) * (xi - xj)**2)
                * (-1) * (l**2 - (xi - xj)**2)/l**4
            )

        # Return the kernel matrix value
        return kij
    
    def plot_results(self):
        """Plots the results of the Gaussian Process model."""

        # Get values
        X = self.X
        y = self.y
        X_indu = self.X_indu
        y_indu = self.y_indu
        y_indu_dx = self.y_indu_dx
        y_indu_dx2 = self.y_indu_dx2

        # Set up figure
        fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        plt.ion()
        plt.show()

        # Plot the data
        ax[0].plot(X, y, 'k.', label='Data')
        ax[0].plot(X_indu, y_indu, 'r-', label='Inferred mean')
        ax[0].set_ylabel('Data')
        ax[0].legend()

        # Plot the acceleration
        ax[1].plot(X_indu, y_indu_dx2, 'r-', label='Inferred acceleration')
        ax[1].fill_between(
            X_indu.flatten(), 
            y_indu_dx2.flatten(), 
            0, where=y_indu_dx2.flatten() >= 0, 
            facecolor='green', 
            interpolate=True,
            label='Positive acceleration'
        )
        ax[1].fill_between(
            X_indu.flatten(), 
            y_indu_dx2.flatten(), 
            0, where=y_indu_dx2.flatten() <= 0, 
            facecolor='red', 
            interpolate=True,
            label='Negative acceleration'
        )
        ax[1].set_ylabel('Acceleration')
        ax[1].set_xlabel('Time')
        ax[1].legend()

        # Return the figure
        return fig, ax
    

