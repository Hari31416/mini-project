import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class CenterExtracter:
    """
    Class containing everything needed to extract the center of the drop from a given image.
    """

    def __init__(self, region=(700, 350, 1200, 700), m=0.466667, c=152) -> None:
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

    def _update_params_(self, key, value):
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

    def _read_image_(self, image_path, output="np"):
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

    def _convert_to_array_(self, image):
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

    def _new_c_(self, x, y):
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

    def _restrict_to_region_(self, image):
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

    def _crop_(self, image, x, y, h, w):
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
        self._restrict_to_region_(image)
        if isinstance(image, Image.Image):
            self.image_final = self.image(x, y, x + w, y + h)
            return self.image_final
        else:
            self.image_final = self.image[y : y + h, x : x + w]
            return self.image_final

    def _get_line_(self, x, y):
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
        m = self._new_c_(x, y)
        x_points = np.arange(0, self.image.shape[1])
        y_line = m * x_points + self.c
        y_line = y_line.astype(int)
        return y_line

    def _threshold_(
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

    def _stricting_(self, x, y):
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
        y_line = self._get_line_(x, y)
        img_x = self.image_final[: y_line[0]]
        Xs = np.zeros(self.image_final.shape)
        for i in range(self.image_final.shape[0]):
            try:
                Xs[i] = img_x[i]
            except:
                Xs[i] = np.ones(self.image_final.shape[0])
        return Xs

    def _center_x_(self, image, reverse=False):
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

    def _center_y_(self, image, reverse=False):
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

    def _center_and_radius_(
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
        (xl, xr), (yu, yd) = self._center_x_(image, reverse=reverse), self._center_y_(
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

    def _show_image_(self, image, title="", binary=False, threshold=110):
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
            image = self._threshold_(threshold=threshold, image=image)
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray")
        plt.grid(True)
        plt.title(title)
        plt.show()

    def _plot_(self, title=""):
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

    def _subtract_image_(self, image, ref_image="ref_image.jpg"):
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
        ref_image_np = plt.imread(ref_image)
        return np.maximum((ref_image_np / 255.0 - image / 255.0) * 255, 0)

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
            self.image = self._read_image_(image_path, output=output)
        else:
            self.image = image_path
        image = self.image
        # Converting to numpy array
        image = self._convert_to_array_(image)
        # Subtractng the reference image
        if subtract:
            reverse = True
            image = self._subtract_image_(image)
        # Cropping the image
        image = self._crop_(image, x, y, h, w)
        # Thresholding the image
        image = self._threshold_(threshold)
        if strict:
            # Stricting the image to the line
            image = self._stricting_(x, y)
        # calculating center
        center, radius = self._center_and_radius_(
            image, crop_included=crop_included, reverse=reverse
        )
        if plot:
            self._plot_(title)

        return radius, center
