import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import EllipseModel
import peakutils
from scipy.interpolate import interp1d


class CenterExtracter:
    """
    Class containing everything needed to extract the center of the drop from a given image.
    """

    def __init__(
        self, region=(700, 350, 1200, 700), m=0.466667, c=152, ref_image=None
    ) -> None:
        """
        Instantiates the class.

        Parameters
        ----------
        region : tuple
            region of the image to be used
        m : float
            slope of the line
        c : float
            y-intercept of the line

        Returns
        -------
        None
        """
        self.region = region
        self.X = region[0]
        self.Y = region[1]
        self.x = 0
        self.y = 0
        self.m = m
        self.c = c
        self.image = None
        self.image_final = None
        self.ref_image = ref_image

    def _update_params(self, key, value):
        """
        Updates the parameters of the class.

        Parameters
        ----------
        key : str
            name of the parameter to be updated
        value : int or float
            value of the parameter to be updated

        Returns
        -------
        None
        """
        setattr(self, key, value)

    def _read_image(self, image_path, output="np"):
        """
        Reads the image from the given path.

        Parameters
        ----------
        image_path : str
            path to the image
        output : str
            output format of the image. Can be either "np" or "PIL"

        Returns
        -------
        PIL.Image or numpy.ndarray
            image read from the given path
        """
        if output.lower() == "np":
            self.image = np.array(plt.imread(image_path))
            return self.image
        elif output.lower() == "pil":
            self.image = Image.open(image_path)
            return self.image

    def _convert_to_array(self, image):
        """
        Converts the image to a numpy array.

        Parameters
        ----------
        image : PIL.Image or numpy.ndarray
            Image to be converted

        Returns
        -------
        numpy.ndarray
            converted image
        """
        if isinstance(image, Image.Image):
            return np.array(image)
        else:
            return image

    def _new_c(self, x, y):
        """
        Calculates the new c value based on the x and y values of the center.

        Parameters
        ----------
        x : int
            x coordinate of the crop
        y : int
            y coordinate of the crop

        Returns
        -------
        int
            new c value
        """
        return self.m * x + self.c - y

    def _restrict_to_region(self, image):
        """
        Restricts the image to the region defined in the constructor.

        Parameters
        ----------
        image : PIL.Image or numpy.ndarray
            Image to be restricted

        Returns
        -------
        PIL.Image or numpy.ndarray
            restricted image
        """
        if isinstance(image, Image.Image):
            self.image = image.crop(self.region)
            return self.image
        else:
            self.image = image[
                self.region[1] : self.region[3],
                self.region[0] : self.region[2],
            ]
            return self.image

    def _crop(self, image, x, y, h, w):
        """
        Crops the image to the region defined in the constructor.

        Parameters
        ----------
        image : PIL.Image or numpy.ndarray
            Image to be cropped
        x : int
            x coordinate of the crop
        y : int
            y coordinate of the crop
        h : int
            height of the crop
        w : int
            width of the crop

        Returns
        -------
        PIL.Image or numpy.ndarray
            cropped image
        """
        self.x = x
        self.y = y
        self._restrict_to_region(image)
        if isinstance(image, Image.Image):
            self.image_final = self.image(x, y, x + w, y + h)
            return self.image_final
        else:
            self.image_final = self.image[y : y + h, x : x + w]
            return self.image_final

    def _get_line(self, x, y):
        """
        Gets the equation of line holding the thin film given the x and y coordinates of the crop

        Parameters
        ----------
        x : int
            x coordinate of the crop
        y : int
            y coordinate of the crop

        Returns
        -------
        numpy.ndarray
            equation of the line
        """
        m = self._new_c(x, y)
        x_points = np.arange(0, self.image.shape[1])
        y_line = m * x_points + self.c
        y_line = y_line.astype(int)
        return y_line

    def _threshold(
        self,
        threshold=110,
        image=None,
    ):
        """
        Thresholds the image.

        Parameters
        ----------
        threshold : float
            threshold value
        image : numpy.ndarray
            image to be thresholded if `None` is given, the image from the constructor is used

        Returns
        -------
        numpy.ndarray
            thresholded image
        """
        if image is None:
            self.image_final = (self.image_final > threshold).astype(int)
            return self.image_final
        else:
            return (image > threshold).astype(int)

    def _stricting(self, x, y):
        """
        Stricts the image to the line defined by the line made by the thin film holder.

        Parameters
        ----------
        x : int
            x coordinate of the crop
        y : int
            y coordinate of the crop

        Returns
        -------
        numpy.ndarray
            strict image
        """
        y_line = self._get_line(x, y)
        img_x = self.image_final[: y_line[0]]
        Xs = np.zeros(self.image_final.shape)
        for i in range(self.image_final.shape[0]):
            try:
                Xs[i] = img_x[i]
            except:
                Xs[i] = np.ones(self.image_final.shape[0])
        return Xs

    def _center_x(self, image, reverse=False):
        """
        Calculates the left and right x coordinates of the box binding the drop

        Parameters
        ----------
        image : numpy.ndarray
            Image to be processed

        Returns
        -------
        xl : int
            left x coordinate of the box binding the drop
        xr : int
            right x coordinate of the box binding the drop

        """
        if reverse:
            xs = image.argmax(axis=0)
        else:
            xs = image.argmin(axis=0)
        xss = np.nonzero(xs)
        xl = xss[0][0]
        xr = xss[0][-1]
        self.centers_x = (xl, xr)
        return xl, xr

    def _center_y(self, image, reverse=False):
        """
        Calculates the up and down y coordinates of the box binding the drop

        Parameters
        ----------
        image : numpy.ndarray
            Image to be processed

        Returns
        -------
        yu : int
            upper y coordinate of the box binding the drop
        yd : int
            lower y coordinate of the box binding the drop
        """
        if reverse:
            ys = image.argmax(axis=1)
        else:
            ys = image.argmin(axis=1)
        yss = np.nonzero(ys)
        yu = yss[0][0]
        yd = yss[0][-1]
        self.centers_y = (yu, yd)
        return yu, yd

    def _center_and_radius(
        self,
        image,
        reverse,
        crop_included=True,
    ):
        """
        Calculates the center of the drop.

        Parameters
        ----------
        image : numpy.ndarray
            Image to be processed

        Returns
        -------
        (x, y) : tuple
            center of the drop
        r : int
            radius of the drop
        """
        (xl, xr), (yu, yd) = self._center_x(image, reverse=reverse), self._center_y(
            image, reverse=reverse
        )

        r1 = int(np.abs((xl - xr)) / 2)
        r2 = int(np.abs((yu - yd)) / 2)
        r = int((r1 + r2) / 2)
        if crop_included:
            x = int(self.X + self.x + xl + r)
            y = int(self.Y + self.y + yu + r)
        else:
            x = int(self.x + xl + r)
            y = int(self.y + yu + r)
        self.center = (x, y)
        self.radii = (r1, r2)
        return (x, y), (r1, r2)

    def _show_image(self, image, title="", binary=False, threshold=110):
        """
        Shows the image.

        Parameters
        ----------
        image : PIL.Image or numpy.ndarray
            Image to be shown
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        if binary:
            image = self._threshold(threshold=threshold, image=image)
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray")
        plt.grid(True)
        plt.title(title)
        plt.show()

    def _plot(self, title=""):
        """
        Plots the image.

        Parameters
        ----------
        title : str
            title of the plot

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image, cmap="gray")
        plt.hlines(-self.Y + self.center[1], 0, self.image.shape[1], color="g")
        plt.vlines(-self.X + self.center[0], 0, self.image.shape[0], color="b")
        plt.annotate(
            f"{(self.center[0], self.center[1])}",
            xy=(-self.X + self.center[0], -self.Y + self.center[1]),
            xytext=(-self.X + self.center[0] + 20, -self.Y + self.center[1] + 30),
            color="r",
        )
        plt.grid()
        plt.title(title)
        plt.show()

    def _subtract_image(self, image):
        """
        Subtracts the reference image from the image.

        Parameters
        ----------
        ref_image : numpy.ndarray
            reference image
        image : numpy.ndarray
            image to be subtracted

        Returns
        -------
        numpy.ndarray
            subtracted image
        """
        ref_image_np = plt.imread(self.ref_image)
        return np.maximum((ref_image_np / 255.0 - image / 255.0) * 255, 0)

    def _all_points_binary(self, img, threshold=110):
        """
        Returns all the points  of the drop.

        Parameters
        ----------
        image : numpy.ndarray
            Image to be processed
        threshold : int
            threshold of the image

        Returns
        -------
        xs : numpy.ndarray
            x coordinates of the points
        ys : numpy.ndarray
            y coordinates of the points
        """
        img = self._threshold(image=img, threshold=threshold)
        ys1 = np.nonzero(img.argmax(axis=1))[0]
        xs1 = np.nonzero(img.argmax(axis=0))[0]
        xs2 = img.argmax(axis=1)[ys1]
        ys2 = img.argmax(axis=0)[xs1]

        xs = np.concatenate((xs2, xs1), axis=0)
        ys = np.concatenate((ys1, ys2), axis=0)
        return xs, ys

    def _get_edges(
        self, array, kernel="gaussian_y", x=None, y=None, thres=0.6, min_array_value=110
    ):
        """
        Gets the coordinates for the edge of a row or column of the image array

        Parameters
        ----------
        array : numpy.ndarray
            image array
        kernel : str
            kernel to be used for the edge detection.
            Available kernels:
                "gaussian" : gaussian filter
                "diff" : differential filter
                "y": No filters, just using interpolation
                "gaussian_y": gaussian filter and interpolation
                "all" : All three kernels
        x : int
            column index of the array
        y : int
            row index of the array
        thres : float
            threshold for the edge detection, used in the `peakutils` class
        min_array_value : int
            minimum value of the array to be considered as an part of the drop.

        Returns
        -------
        points: tuple of tuples
            coordinates of the left (and possibly right) edge of the drop
        """
        # Ignoring the noise
        if max(array) < min_array_value:
            return None

        # Interpolating the array
        inter = interp1d(
            range(len(array)), array, kind="cubic", bounds_error=False, fill_value=0
        )
        X = np.arange(0, len(array), 0.1)
        Y = inter(X)

        # Convolving with the gaussian kernel and Differential Kernel
        diff_kernel = [-1, 0, 1]
        gaussian = np.arange(-6, 6, 0.5)
        gaussian_kernel = np.exp(-(gaussian ** 2) / 2.0) / np.sqrt(2 * np.pi)
        gaussian = np.convolve(Y, gaussian_kernel, mode="same")
        differentiated = np.convolve(gaussian, diff_kernel, mode="same")

        # Detecting the peaks
        gaussian_points = peakutils.indexes(gaussian, thres=thres) / 10
        diff_points = peakutils.indexes(-differentiated, thres=thres) / 10
        y_points = peakutils.indexes(Y, thres=thres) / 10

        # Compenseting for more than two peaks
        peaks = [gaussian_points, diff_points, y_points]
        modified_peaks = []
        for peak in peaks:
            if len(peak) > 2:
                peak = np.array([peak[0], peak[-1]])
            else:
                peak = peak
            modified_peaks.append(peak)
        gaussian_points, diff_points, y_points = modified_peaks
        # Using the method specified
        if kernel.lower() == "all":
            point = (gaussian_points + diff_points + y_points) / 3
        elif kernel.lower() == "gaussian":
            point = gaussian_points
        elif kernel.lower() == "diff":
            point = diff_points
        elif kernel.lower() == "y":
            point = y_points
        elif kernel.lower() == "gaussian_y":
            point = (gaussian_points + y_points) / 2
        else:
            raise ValueError(
                "Invalid kernel. Available kernels: 'gaussian', 'diff', 'y', 'all', 'gaussian_y'"
            )

        # Getting the x and y coordinates
        if x is None:
            point1 = (round(point[0], 1), y)
            try:
                point2 = (round(point[1], 1), y)
            except IndexError:
                point2 = (round(point[0], 1), y)
        else:
            point1 = (x, round(point[0], 1))
            try:
                point2 = (x, round(point[1], 1))
            except IndexError:
                point2 = (x, round(point[0], 1))
        return point1, point2

    def _get_points(self, img, kernel="gaussian_y", thres=0.6, min_array_value=50):
        """
        Uses the `get_edges` function to get the coordinates for the edge of the drop

        Parameters
        ----------
        img : numpy.ndarray
            Image to be processed
        method : str
            method to be used for the edge detection.
            Available methods:
                "gaussian" : gaussian filter
                "diff" : differential filter
                "y": No filters, just using interpolation
                "gaussian_y": gaussian filter and interpolation
                "all" : All three methods
        thres : float
            threshold for the edge detection, used in the `peakutils` class
        min_array_value : int
            minimum value of the array to be considered as an part of the drop.

        Returns
        -------
        xs : numpy.ndarray
            x coordinates of the points
        ys : numpy.ndarray
            y coordinates of the points

        """
        all_points_1 = []
        all_points_2 = []
        for i in range(img.shape[0]):
            points1 = self._get_edges(
                img[i, :],
                kernel=kernel,
                y=i,
                thres=thres,
                min_array_value=min_array_value,
            )
            if points1 is not None:
                all_points_1.append(points1)

        for i in range(img.shape[1]):
            points2 = self._get_edges(
                img[:, i],
                kernel=kernel,
                x=i,
                thres=thres,
                min_array_value=min_array_value,
            )
            if points2 is not None:
                all_points_2.append(points2)

        points = []
        for i in range(len(all_points_1)):
            (xl, y), (xr, y) = all_points_1[i]
            points.append((xl, y))
            points.append((xr, y))

        for i in range(len(all_points_2)):
            (x, yl), (x, yr) = all_points_2[i]
            points.append((x, yl))
            points.append((x, yr))

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        return xs, ys

    def _drop_points(self, xs, ys, drop_points=5):
        """
        Drops some of the points which are usually noisey

        Parameters
        ----------
        xs : numpy.ndarray
            x coordinates of the points
        ys : numpy.ndarray
            y coordinates of the points
        drop_points : int
            number of points to be dropped

        Returns
        -------
        xs : numpy.ndarray
            x coordinates of the points
        ys : numpy.ndarray
            y coordinates of the points
        """
        x_sort_index = np.argsort(xs)
        values = x_sort_index[drop_points:-drop_points]
        xs = xs[values]
        ys = ys[values]

        y_sort_index = np.argsort(ys)
        values = y_sort_index[drop_points:-drop_points]
        xs = xs[values]
        ys = ys[values]
        return xs, ys

    def _all_points_grayscale(
        self,
        img,
        thres=0.6,
        min_array_value=50,
        plot=False,
        kernel="gaussian_y",
        drop_points=5,
    ):
        """
        Gets all the points of the circumference of the drop using the functions defined above.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be processed
        thres : float
            threshold for the edge detection, used in the `peakutils` class
        min_array_value : int
            minimum value of the array to be considered as an part of the drop.
        plot: bool
            if True, the image is plotted
        kernel : str
            kernel to be used for the edge detection.
            Available kernels:
                "gaussian" : gaussian filter
                "diff" : differential filter
                "y": No filters, just using interpolation
                "gaussian_y": gaussian filter and interpolation
                "all" : All three kernels
        drop_points : int
            number of points to be dropped


        Returns
        -------
        xs : numpy.ndarray
            x coordinates of the points
        ys : numpy.ndarray
            y coordinates of the points
        """
        xs, ys = self._get_points(
            img, kernel=kernel, thres=thres, min_array_value=min_array_value
        )
        xs, ys = self._drop_points(xs, ys, drop_points=drop_points)
        return xs, ys

    def fit_ellipse(
        self,
        image,
        crop_included=True,
        binary=False,
        x=0,
        y=0,
        h=500,
        w=500,
        subtract=True,
        plot=False,
        title="",
        drop_points=5,
        kernel="gaussian_y",
        min_array_value=110,
        threshold=110,
    ):
        """
        Fits an ellipse to the drop.

        Parameters
        ----------
        image : `str` or numpy.ndarray
            Image to be processed
        crop_included : bool
            if True, the crop is included in the center of the drop
        binary : bool
            if True, the image is binarized
        x : int
            x coordinate to use to crop
        y : int
            y coordinate to use to crop
        h : int
            height to use to crop
        w : int
            width to use to crop
        subtract : bool
            if True, the image is subtracted from the reference image
        plot : bool
            if True, the image is plotted
        title : str
            title of the plot
        drop_points : int
            number of points to be dropped while getting the points of the circumference
        kernel : str
            kernel to be used for the edge detection.
            Available kernels:
            "gaussian" : gaussian filter
            "diff" : differential filter
            "y": No filters, just using interpolation
            "gaussian_y": gaussian filter and interpolation
            "all" : All three kernels
        min_array_value : int
            minimum value of the array to be considered as an part of the drop.

        Returns
        -------
        (x, y) : tuple
            center of the drop
        (r1, r2) : tuple
            radii of the drop
        theta : float
            angle of the drop
        """
        img = self._read_image(image_path=image)
        if subtract:
            img = self._subtract_image(img)
        img = self._crop(img, x, y, h, w)
        if binary:
            xs, ys = self._all_points_binary(img, threshold=threshold)
        else:
            xs, ys = self._all_points_grayscale(
                img,
                thres=0.6,
                kernel=kernel,
                drop_points=drop_points,
                min_array_value=min_array_value,
            )
        points = np.array([xs, ys]).T
        ell = EllipseModel()
        ell.estimate(points)
        xc, yc, a, b, theta = ell.params

        if crop_included:
            xc += self.X
            yc += self.Y
        self.center = (int(xc), int(yc))
        self.r1 = int(a)
        self.r2 = int(b)

        if plot:
            t = np.linspace(0, 2 * np.pi, 100)
            plt.imshow(self.image_final, cmap="gray")
            plt.plot(xc - self.X + a * np.cos(t), yc - self.Y + b * np.sin(t))
            plt.grid(color="lightgray", linestyle="--")
            plt.scatter(-self.X + self.center[0], -self.Y + self.center[1], color="g")
            plt.scatter(xs, ys, s=5, c="r")
            plt.xlim([np.min(xs) - 5, np.max(xs) + 5])
            plt.ylim([np.min(ys) - 5, np.max(ys) + 5])
            plt.annotate(
                f"{(self.center[0], self.center[1])}",
                xy=(-self.X + self.center[0], -self.Y + self.center[1]),
                xytext=(-self.X + self.center[0], -self.Y + self.center[1]),
                color="r",
            )
            plt.title(title)
            plt.show()
        return self.center, (self.r1, self.r2), round(theta, 2)

    def get_center(
        self,
        image_path,
        subtract=False,
        x=0,
        y=0,
        h=50,
        w=50,
        output="np",
        title="",
        strict=True,
        plot=True,
        threshold=110,
        crop_included=True,
        reverse=False,
    ):
        """
        Returns the center of the drop.

        Parameters
        ----------
        image_path : str
            path to the image
        x : int
            x coordinate of the crop
        y : int
            y coordinate of the crop
        h : int
            height of the crop
        w : int
            width of the crop
        output : str
            format to use. Can be "np" or "pil"
        title : str
            title of the plot
        strict : bool
            if `True` the image is strict to the line defined by the thin film holder
        plot : bool
            if `True` the image is plotted
        threshold : float
            threshold value

        Returns
        -------
        (x, y) : tuple
            center of the drop
        """
        # Reading image
        if isinstance(image_path, str):
            self.image = self._read_image(image_path, output=output)
        else:
            self.image = image_path
        image = self.image
        # Converting to numpy array
        image = self._convert_to_array(image)
        # Subtractng the reference image
        if subtract:
            reverse = True
            image = self._subtract_image(image)
        # Cropping the image
        image = self._crop(image, x, y, h, w)
        # Thresholding the image
        image = self._threshold(threshold)
        if strict:
            # Stricting the image to the line
            image = self._stricting(x, y)
        # calculating center
        center, radii = self._center_and_radius(
            image, crop_included=crop_included, reverse=reverse
        )
        if plot:
            self._plot(title)

        return radii, center
