import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as sg
import h5py
import datetime as dt
import matplotlib.transforms as transforms

from scipy import stats
from matplotlib.patches import Ellipse
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


sns.set()
sns.set_style("whitegrid")

"""
Code source: https://matplotlib.org/3.1.1/gallery/statistics
/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

Theory: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
"""
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Reading in data from MAT file

datadir = "../data/"
with h5py.File(datadir + "m1244.mat",'r') as file:
    # Looking at contents of file
    print(file.keys())
    # Finding the variables contained in the key
    variables = list(file.get("m1244"))
    print(variables)

    # Storing the variables in a dictionary
    data = {}
    for i in variables:
        data[i] = np.array(file.get(f"m1244/{i}")).squeeze()  # removes excessive dims

    # Convert a complex tuple (real, imag) to a complex number a + jb
    cv = np.zeros((data["cv"].shape[0], data["cv"].shape[1]), dtype=complex)
    for i in range(len(data["cv"])):
        for j in range(len(data["cv"][0])):
            cv[i][j] = data["cv"][i][j][0] + 1j*data["cv"][i][j][1]
    data["cv"] = cv

    # Convert matlab datenum format to a datetime object for aesthetics
    num = []
    for i in data["num"]:
        num.append(dt.datetime.fromordinal(int(i)) + dt.timedelta(days=i%1) - dt.timedelta(days = 366))
    data["num"] = np.array(num)

# Reading data done. File closes automatically with "with ..."


# Looking at deepest mooring
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(data["num"], data["cv"][3].real)
ax.plot(data["num"], data["cv"][3].imag)
fig.tight_layout()


# Rotating by the mean flow
phi = np.angle(np.mean(data["cv"][3]))
cv_rotated = data["cv"]*np.exp(-1j*phi)

window = sg.windows.hann(24)
# Smoothing data using a hann window with window size 24
cv_smoothed = sg.convolve(cv_rotated[3], window, mode="same")/np.sum(window)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(data["num"], cv_smoothed.real)
ax.plot(data["num"], cv_smoothed.imag)
fig.tight_layout()

# Creating 2D histogram

# Defining bins which in this case are equal
ubins = np.arange(-50, 50, 1)
vbins = np.arange(-50, 50, 1)

# Calculating histogram and bin centers
H = stats.binned_statistic_2d(cv_rotated[3].real, cv_rotated[3].imag, None, bins=[ubins, vbins], statistic="count")
ucenter = 0.5*(H.x_edge[1:] + H.x_edge[:-1])
vcenter = 0.5*(H.y_edge[1:] + H.y_edge[:-1])

U, V = np.meshgrid(ucenter, vcenter, indexing="ij")


fig, ax = plt.subplots(1, 1, figsize=(10, 8))

cmap = plt.cm.get_cmap("jet", 60)
im = ax.pcolormesh(U, V, H.statistic,
                    vmin=0,
                    vmax=45,
                    cmap=cmap,
                    shading="flat",
                    )
fig.colorbar(im, ax=ax)
ax.plot(np.mean(cv_rotated[3]).real, np.mean(cv_rotated[3]).imag, "wo", markerfacecolor="k")
ax.axhline(0, linestyle=":", color="white")
ax.axvline(0, linestyle=":", color="white")
ax.set_aspect("equal")

confidence_ellipse(data["cv"][3].real, data["cv"][3].imag, ax=ax, n_std=1, edgecolor="black")

plt.show()
