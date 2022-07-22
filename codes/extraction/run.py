import os
import matplotlib.pyplot as plt
from extract_center import CenterExtracter
import pandas as pd
import tqdm
import numpy as np

plt.rcdefaults()


class Run:
    def __init__(self, path=os.curdir, save_path=os.curdir, **centerextracter):
        """
        The class uses all the three methods developed to extract centers of the drops from images.

        Parameters
        ----------
        path : str
            The path of the folder containing the images.

        """
        self.path = path
        self.save_path = save_path
        self.c = CenterExtracter(**centerextracter)
        self.centers_dc = {}
        self.radii_dc = {}
        self.centers_si = {}
        self.radii_si = {}
        self.centers_ap = {}
        self.radii_ap = {}
        self.thetas_ap = {}
        self.ref_image = None

    def rename_images(self):
        """
        Renaming the images to make them unique.
        """
        images = os.listdir(
            self.path,
        )
        for i, image in enumerate(images):
            extension = image.split(".")[-1]
            os.rename(
                os.path.join(self.path, image),
                os.path.join(self.path, f"{i}.{extension}"),
            )

    def get_images(
        self,
    ):
        """
        Gets a list of all the images in the directory.

        Parameters
        ----------
        extensions : list
            The extensions of the images to be processed.

        Returns
        -------
        images : list
            The list of images to be processed.

        """
        # Getting list of images
        images = os.listdir(self.path)
        images = [img for img in images if img.endswith(".jpg")]
        images = sorted(images, key=lambda x: int(x.split(".")[0]))
        images = [f"{self.path}/{image}" for image in images]
        return images

    def _set_ref_image(self, img_num=-10):
        """
        Sets the reference image for the `CenterExtracter` class.
        """
        images = self.get_images()
        last_image = images[img_num]
        self.c.ref_image = last_image
        self.ref_image = last_image

    def save_to_csv(self, method="ap", file_name=None, include_crop=False):
        """
        Writes the centers and radii to a csv file.

        Parameters
        ----------
        method : str
            The method to be used for the extraction.
            available methods:
                1. dynamic_cropping (dc)
                2. subtracting_images (si)
                3. all_poins (ap)
        file_name : str
            The path of the file to be written.
        include_crop : bool
            Whether to include the crop coordinates in the csv file. Should be used when the method is "dc"

        Returns
        -------
        df : pandas.DataFrame
            The dataframe containing the centers and radii.
        """
        if file_name is None:
            file_name = f"centers_using_{method}.csv"
        else:
            if not file_name.endswith(".csv"):
                file_name = file_name + ".csv"

        method_dict = {
            "dc": (self.centers_dc, self.radii_dc),
            "si": (self.centers_si, self.radii_si),
            "ap": (self.centers_ap, self.radii_ap),
        }
        if method not in method_dict.keys():
            raise ValueError(f"{method} is not a valid method")

        centers, radii = method_dict[method]
        df1 = pd.DataFrame.from_dict(
            centers,
        ).T.reset_index()
        df1.columns = ["id", "x", "y"]
        # Compensating for the crop
        if include_crop:
            df1["x"] = df1["x"] + self.c.X
            df1["y"] = df1["y"] + self.c.Y

        df2 = pd.DataFrame.from_dict(
            radii,
        ).T.reset_index()
        df2.columns = ["id", "r1", "r2"]

        if method == "ap":
            df3 = pd.DataFrame(self.thetas_ap, index=[0]).T.reset_index()
            df3.columns = ["id", "theta"]
            df = pd.merge(df1, df2, on="id")
            df = pd.merge(df, df3, on="id")
        else:
            df = pd.merge(df1, df2, on="id")

        df.to_csv(os.path.join(self.save_path, file_name), index=False)
        return df

    def dynamic_cropping_one_step(
        self,
        x=150,
        y=40,
        h=40,
        w=40,
        start=0,
        end=-1,
        **kwargs,
    ):
        """
        Uses dynamic cropping method to extract centers of the drops.

        Parameters
        ----------
        x : int
            The x-coordinate of the crop
        y : int
            The y-coordinate of the crop
        h : int
            The height of the crop
        w : int
            The width of the crop
        start : int
            The starting index of the images to be processed
        end : int
            The ending index of the images to be processed
        kwargs : dict
            The keyword arguments to be passed to the `get_center` method.

        Returns
        -------
        centers_dc : dict
            The dictionary containing the centers of the drops.
        radii_dc : dict
            The dictionary containing the radii of the drops.
        """
        images = self.get_images()
        (r1, r2), (xc_p, yc_p) = self.c.get_center(
            images[start],
            x=x,
            y=y,
            h=h,
            w=w,
            **kwargs,
        )
        self.centers_dc[images[start]] = (xc_p, yc_p)
        self.radii_dc[images[start]] = (r1, r2)

        for img in images[start + 1 : end]:
            try:
                (r1, r2), (xc, yc) = self.c.get_center(
                    img, x=x, y=y, h=h, w=h, **kwargs
                )
                self.centers_dc[img] = (xc_p, yc_p)
                self.radii_dc[img] = (r1, r2)
                x = x + (xc - xc_p)
                y = y + (yc - yc_p)
                xc_p = xc
                yc_p = yc
            except:
                print("Error at: " + img)
                continue
        return self.centers_dc, self.radii_dc

    def dynamic_cropping(
        self,
        crop_1=(150, 40, 40, 40),
        crop_2=(120, 250, 50, 50),
        crop_3=(250, 230, 40, 40),
        positions_1=(0, 70),
        positions_2=(75, 100),
        positions_3=(106, 155),
        save=True,
        file_name=None,
        **kwargs,
    ):
        """
        Uses dynamic cropping method to extract centers of the drops.

        Parameters
        ----------
        steps : int
            The number of steps to be taken in the dynamic cropping process.
        crop_1 : tuple
            The crop parameters for the first step.
        crop_2 : tuple
            The crop parameters for the second step.
        crop_3 : tuple
            The crop parameters for the third step.
        kwargs : dict
            The keyword arguments to be passed to the `get_center` method.

        Returns
        -------
        df : pandas.DataFrame
            The dataframe containing the centers and radii. (If save=True) else
        centers_si : dict
            The dictionary containing the centers of the drops.
        radii_si : dict
            The dictionary containing the radii of the drops.
        """
        print("Starting dynamic cropping...")
        print("First step...")
        self.dynamic_cropping_one_step(*crop_1, *positions_1, **kwargs)
        print("Second step...")
        self.dynamic_cropping_one_step(*crop_2, *positions_2, **kwargs)
        print("Third step...")
        self.dynamic_cropping_one_step(*crop_3, *positions_3, **kwargs)
        if save:
            print("Saving to csv...")
            df = self.save_to_csv(method="dc", file_name=file_name, include_crop=True)
            print("Done!")
            return df
        else:
            return self.centers_si, self.radii_si

    def subtracting_images(self, h=500, w=500, save=True, file_name=None, **kwargs):
        """
        Uses subtracting images method to extract centers of the drops.

        Parameters
        ----------
        h : int
            The height of the crop.
        w : int
            The width of the crop.
        save : bool
            Whether to save the centers and radii to a csv file.
        kwargs : dict
            The keyword arguments to be passed to the `get_center` method.

        Returns
        -------
        df : pandas.DataFrame
            The dataframe containing the centers and radii. (If save=True) else
        centers_si : dict
            The dictionary containing the centers of the drops.
        radii_si : dict
            The dictionary containing the radii of the drops.
        """
        print("Getting list of images...")
        images = self.get_images()
        print("Extracting data from images...")
        for img in tqdm.tqdm(images, desc="Extracting data from images"):
            if img.endswith(".jpg"):
                try:
                    (r1, r2), (x, y) = self.c.get_center(img, h=500, w=500, **kwargs)
                    self.centers_si[img] = (x, y)
                    self.radii_si[img] = (r1, r2)
                except:
                    print("Error on: ", img)
                    self.centers_si[img] = (None, None)
                    self.radii_si[img] = (None, None)
                    continue
        if save:
            print("Saving to csv...")
            df = self.save_to_csv(method="si", file_name=file_name, include_crop=False)
            print("Done!")
            return df
        else:
            return self.centers_si, self.radii_si

    def all_points(
        self,
        save=True,
        method="ap",
        file_name=None,
        raise_error=False,
        num_images=None,
        verbose=True,
        dynamic_cropping=False,
        **kwargs,
    ):
        """
        Uses subtracting images method to extract centers of the drops.

        Parameters
        ----------
        h : int
            The height of the crop.
        w : int
            The width of the crop.
        save : bool
            Whether to save the centers and radii to a csv file.
        kwargs : dict
            The keyword arguments to be passed to the `fit_ellipse` method.

        Returns
        -------
        df : pandas.DataFrame
            The dataframe containing the centers and radii. (If save=True) else
        centers_ap : dict
            The dictionary containing the centers of the drops.
        radii_ap : dict
            The dictionary containing the radii of the drops.
        theta_ap : dict
            The dictionary containing the theta of the drops.
        """
        print("Getting list of images...")
        images = self.get_images()
        print("Extracting data from images...")
        if num_images is not None:
            images = images[:num_images]
        if dynamic_cropping:
            img = images[0]
            (x, y), (a, b), theta = self.c.fit_ellipse(img, **kwargs)
            self.centers_ap[img] = (x, y)
            self.radii_ap[img] = (a, b)
            self.thetas_ap[img] = theta * 180 / np.pi
        for img in tqdm.tqdm(images, desc="Extracting data from images..."):
            if img.endswith(".jpg"):
                if dynamic_cropping:
                    x_new = x - 25
                    y_new = y - 25
                    h = 50
                    region_of_interest = (x_new, y_new, x_new + h, y_new + h)
                    self.c = CenterExtracter(
                        region=region_of_interest, ref_image=self.ref_image
                    )
                try:
                    # print(str(img))
                    (x, y), (a, b), theta = self.c.fit_ellipse(img, **kwargs)
                    self.centers_ap[img] = (x, y)
                    self.radii_ap[img] = (a, b)
                    self.thetas_ap[img] = theta * 180 / np.pi
                except Exception as e:
                    if raise_error:
                        raise e
                    else:
                        if verbose:
                            print("Error on: ", img)
                        self.centers_ap[img] = (None, None)
                        self.radii_ap[img] = (None, None)
                        self.thetas_ap[img] = None
                        continue
        if save:
            print("Saving to csv...")
            df = self.save_to_csv(
                method=method, file_name=file_name, include_crop=False
            )
            print("Done!")
            return df
        else:
            return self.centers_si, self.radii_si, self.thetas_ap
