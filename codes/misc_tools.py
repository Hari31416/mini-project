import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from math import factorial
import matplotlib

plt.rcdefaults()
#


class Smoother:
    """
    A class to deal with the smoothing of the data extracted
    """

    def __init__(self, df) -> None:
        """
        Initializes the class

        Parameters
        ----------
        df : DataFrame or string
            The dataframe to be smoothed or the path to the file
        """
        self.df = df
        self.smooth_df = None

    def savitzky_golay(self, y, window_size, order=3, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
        """

        try:
            window_size = np.abs(int(window_size))
            order = np.abs(int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat(
            [
                [k ** i for i in order_range]
                for k in range(-half_window, half_window + 1)
            ]
        )
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode="valid")

    def moving_average(self, x, w):
        """
        Calculates the moving average

        Parameters
        ----------
        x : array_like
            The data to be smoothed
        w : int
            The window size

        Returns
        -------
        y : array_like
            The smoothed data

        """
        return np.convolve(x, np.ones(w), "valid") / w

    def ema(self, x, w):
        """
        Calculates the exponential moving average

        Parameters
        ----------
        x : array_like
            The data to be smoothed
        w : int
            The window size

        Returns
        -------
        y : array_like
            The smoothed data
        """
        temp_df = pd.DataFrame(x).reset_index()
        temp_df.columns = ["x"]
        ema = temp_df.y.ewm(span=w).mean()
        return ema

    def remove_noises(self, df, row="r", min_val=8, max_val=13):
        """
        Removes noisefull data using the unrealistic values of r.

        Parameters
        ----------
        df : DataFrame
            The dataframe to be cleaned
        row : str
            The name of the row to be cleaned
        min_val : int
            The minimum value of the column to be considered as noise
        max_val : int
            The maximum value of the column to be considered as noise

        Returns
        -------
        df : DataFrame
            The cleaned dataframe
        """
        return df[(df[row] < max_val) & (df[row] > min_val)]

    def preporocess(self, remove_noise=True, **kwargs):
        """
        Does the preprocessing of the data. These processes are:
        - Reads the DataFrame if already not read
        - Remove Null values
        - Remove noise
        - Calculating r and velocity

        Parameters
        ----------
        kwargs : dict
            The arguments to be passed to the `remove_noises` function
        """
        # Remove Null Values
        if isinstance(self.df, str):
            self.df = pd.read_csv(self.df)
            df = self.df.copy()

        df = self.df[self.df["x"].notna()]

        # Getting average radius
        df["r"] = ((df["r1"] + df["r2"]) / 2).astype(int)

        if remove_noise:
            df = self.remove_noises(df, **kwargs)

        # get velocities
        df["vy"] = df["y"].diff()
        df["vx"] = df["x"].diff()
        df["v"] = np.sqrt(df["vy"] ** 2 + df["vx"] ** 2)
        return df

    def smoothen(self, filter="sg", w=43, order=3):
        """
        Smoothens the dataframe using savitzky golay, moving average or exponential moving average

        Parameters
        ----------
        filter : str
            The filter to be used. Can be either "sg" for savitzky golay, "ma" for moving average or\
                 "ema" for exponential moving average
        w : int
            The window size
        order : int
            The order of the polynomial
        
        Returns
        -------
        df : DataFrame
            The smoothened dataframe
        
        """
        df = self.preporocess()

        num_cols = df.columns[(df.dtypes != "object")].values

        if filter == "sg":
            for col in num_cols:
                col_value = df[col].values
                df[col] = self.savitzky_golay(col_value, window_size=w, order=order)

        elif filter == "ema":
            for col in num_cols:
                df[col] = self.ema(df[col], w)
        elif filter == "moving_average":
            for col in num_cols:
                df[col] = self.moving_average(df[col], w)
        self.smooth_df = df
        return df


class Plotter:
    """
    A class to deal with the plot related stuffs
    """

    def __init__(self, df, save_path=os.path.join(os.getcwd(), "plots")):
        """
        Instantiates the Plotting class

        Parameters
        ----------
        df : DataFrame or string
            The dataframe to be plotted or the path to the csv file
        save_path : str
            The path to save the plots
        """
        self.df = df
        self.smoothe_df = None
        self.save_path = save_path
        self.s = Smoother(df)
        os.makedirs(save_path, exist_ok=True)

    def _validate_df(self):
        if isinstance(self.df, str):
            df = pd.read_csv(self.df)
            df = df[df.notna()]
            self.df = df.copy()
        else:
            self.df = self.df[self.df.notna()]
            df = self.df.copy()
        return df

    def get_samples(self, len_samples=49, title="", file_name=None):
        """
        Gets some sample images

        Parameters
        ----------
        len_samples : int
            The number of samples to be taken
        title : str
            The title of the plot
        file_name : str
            The name of the file to be saved

        Returns
        -------
        None
        """

        plt.figure(figsize=(30, 30))
        rows = int(np.sqrt(len_samples))
        df = self._validate_df()
        df = df[df["x"].notna()]
        samples = random.sample(list(df["id"]), len_samples)
        for img in samples:
            plt.subplot(rows, rows, samples.index(img) + 1)
            x, y = (
                df[df["id"] == img]["x"].values[0],
                df[df["id"] == img]["y"].values[0],
            )
            plt.imshow(plt.imread(img)[200:, 500:], cmap="gray")
            plt.hlines(y - 200, 0, 1300 - 600, color="r")
            plt.vlines(x - 500, 0, 800 - 300, color="g")
            subtitle = img.split("/")[-1]
            plt.title(subtitle)
            plt.axis("off")
        # plt.title(title)
        plt.tight_layout()
        plt.annotate(
            title,
            (0, 0),
            (0, -30),
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
            fontsize=30,
        )
        if file_name:
            save_dir = os.path.join(self.save_path, file_name)
            plt.savefig(save_dir)
        plt.show()

    def plot_one(
        self, col1, col2=None, title="", filename=None, smooth=True, scatter=None
    ):
        """
        Plots two columns or one column with time

        Parameters
        ----------
        col1 : str
            The name of the column to be plotted
        col2 : str
            The name of the second column to be plotted. If `None`, the column will be plotted with time
        title : str
            The title of the plot
        filename : str
            The name of the file to be saved
        smooth : bool
            Whether to smooth the data or not

        Returns
        -------
        None
        """
        if smooth:
            if scatter is None:
                scatter = False
            if self.smoothe_df is None:
                self.smoothe_df = self.s.smoothen()
            df = self.smoothe_df.copy()
        else:
            if scatter is None:
                scatter = True
            df = self._validate_df()

        plt.figure(figsize=(10, 10))
        if col2 is None:
            x_values = list(range(len(df[col1])))
            col2 = "time"
        else:
            x_values = df[col2]
        if scatter:
            plt.scatter(x_values, df[col1])
        else:
            plt.plot(x_values, df[col1])

        plt.title(title)
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.show()
        if filename:
            save_dir = os.path.join(self.save_path, filename)
            plt.savefig(save_dir)

    def plot_all(
        self,
        cols=[
            "x",
            "y",
        ],
        title="",
        minus_y=True,
        file_name=None,
        smooth=True,
        scatter=None,
    ):
        """
        Plots two varibales in the dataframe with themselves and with time.

        Parameters
        ----------
        cols : list
            The names of the columns to be plotted (Should be of length 2)
        title : str
            The title of the plot
        minus_y : bool
            Whether to use negative y axis or not
        file_name : str
            The path to save the plot
        smooth : bool
            Whether to smooth the data or not
        """
        font = {"size": 18}
        matplotlib.rc("font", **font)
        if smooth:
            if scatter is None:
                scatter = False
            if self.smoothe_df is None:
                self.smoothe_df = self.s.smoothen()
            df = self.smoothe_df.copy()
        else:
            if scatter is None:
                scatter = True
            if isinstance(self.df, str):
                df = pd.read_csv(self.df)
            else:
                df = self.df.copy()

        if minus_y:
            df["y"] = -df["y"]

        plt.figure(figsize=(16, 16))
        plt.suptitle(title, fontsize=25)
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        if not scatter:
            ax1.plot(df[cols[1]])
        else:
            ax1.scatter(list(range(len(df[cols[1]]))), df[cols[1]])
        ax1.set_title(f"{cols[1]} vs time")
        ax1.set_xlabel("time")
        ax1.set_ylabel(cols[1])

        if not scatter:
            ax2.plot(df[cols[0]])
        else:
            ax2.scatter(list(range(len(df[cols[1]]))), df[cols[0]])
        ax2.set_title(f"{cols[0]} vs time")
        ax2.set_xlabel("time")
        ax2.set_ylabel(cols[0])

        if not scatter:
            ax3.plot(df[cols[0]], df[cols[1]])
        else:
            ax3.scatter(df[cols[0]], df[cols[1]])
        ax3.set_title(f"{cols[0]} vs {cols[1]}")
        ax3.set_xlabel(cols[0])
        ax3.set_ylabel(cols[1])

        if file_name:
            if "." in file_name:
                save_dir = os.path.join(self.save_path, file_name)
                plt.savefig(save_dir)
            else:
                save_dir = os.path.join(self.save_path, f"{file_name}.png")
                plt.savefig(save_dir)
            plt.savefig(file_name)
