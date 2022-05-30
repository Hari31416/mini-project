import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from math import factorial
import matplotlib

plt.rcdefaults()


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
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
        [[k ** i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def ema(x, w):
    temp_df = pd.DataFrame(x).reset_index()
    temp_df.columns = ["x"]
    ema = temp_df.y.ewm(span=w).mean()
    return ema


def remove_noises(df, row="r", min_val=8, max_val=13):
    """
    Removes noisefull data using the unrealistic values of r.


    """
    return df[(df[row] < max_val) & (df[row] > min_val)]


def get_samples(len_samples=49, df_path=None, title="", file_name=None):
    """
    Gets some sample images
    """

    plt.figure(figsize=(30, 30))
    rows = int(np.sqrt(len_samples))
    if isinstance(df_path, str):
        df = pd.read_csv(df_path)
    else:
        df = df_path
    df = df[df["x"].notna()]
    samples = random.sample(list(df["id"][:-10]), len_samples)
    for img in samples:
        plt.subplot(rows, rows, samples.index(img) + 1)
        x, y = (
            df[df["id"] == img]["x"].values[0],
            df[df["id"] == img]["y"].values[0],
        )
        plt.imshow(plt.imread(img)[200:, 500:], cmap="gray")
        plt.hlines(y - 200, 0, 1300 - 600, color="r")
        plt.vlines(x - 500, 0, 800 - 300, color="g")
        plt.title(img)
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
        plt.savefig(file_name)
    plt.show()


def preporocess(df, **kwargs):
    # Remove Null Values
    df = df[df["x"].notna()]

    # Getting average radius
    df["r"] = ((df["r1"] + df["r2"]) / 2).astype(int)

    # Removing 'wrong' values
    df = remove_noises(df, **kwargs)
    # get velocities
    df["vy"] = df["y"].diff()
    df["vx"] = df["x"].diff()
    df["v"] = np.sqrt(df["vy"] ** 2 + df["vx"] ** 2)
    return df


def smoothen(df, filter="sg", w=43, order=3):
    if isinstance(df, str):
        df = pd.read_csv(df)
        df = df.copy()
    else:
        df = df.copy()
    df = preporocess(df)

    num_cols = df.columns[(df.dtypes != "object")].values

    if filter == "sg":
        for col in num_cols:
            col_value = df[col].values
            df[col] = savitzky_golay(col_value, window_size=w, order=order)

    elif filter == "ema":
        for col in num_cols:
            df[col] = ema(df[col], w)
    elif filter == "moving_average":
        for col in num_cols:
            df[col] = moving_average(df[col], w)

    return df


def plot_one(df, col1, col2=None, title=""):
    plt.figure(figsize=(10, 10))
    if col2 is None:
        plt.plot(df[col1])
        col2 = "Time"
    plt.title(title)
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.show()


def plot_all(
    df,
    cols=[
        "x",
        "y",
        "v",
    ],
    title="",
    minus_y=True,
    file_path=None,
):
    font = {"size": 18}
    matplotlib.rc("font", **font)

    if minus_y:
        df["y"] = -df["y"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(title, fontsize=25)
    ax = axes[0][0]
    ax.plot(df[cols[0]], df[cols[1]])
    ax.set_title(f"{cols[0]} vs {cols[1]}")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])

    ax = axes[0][1]
    ax.plot(df[cols[0]])
    ax.set_title(f"{cols[0]} vs time")
    ax.set_xlabel("time")
    ax.set_ylabel(cols[0])

    ax = axes[1][0]
    ax.plot(df[cols[1]])
    ax.set_title(f"{cols[1]} vs time")
    ax.set_xlabel("time")
    ax.set_ylabel(cols[1])

    ax = axes[1][1]
    ax.plot(df[cols[2]])
    ax.set_title(f"{cols[2]} vs time")
    ax.set_xlabel("time")
    ax.set_ylabel(cols[2])
    # fig.subplots_adjust(
    #     left=None,
    #     bottom=None,
    #     right=None,
    #     top=None,
    #     wspace=0.1,
    #     hspace=0.1,
    # )
    if file_path:
        if "." in file_path:
            plt.savefig(file_path)
        else:
            plt.savefig(file_path + ".jpg")
        plt.savefig(file_path)
