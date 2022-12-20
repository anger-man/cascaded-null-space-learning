import numpy.random as rng
import numpy as np
from odl import DiscreteLp
from odl.phantom.geometric import _getshapes_2d, cuboid

def gen_phantom(space, p=0.5):
    """ Generate random phantom in 2D space.
    Choosing (p,1-p) betweend random sheep_logan based phantom (with more rectangles)
    and random febris based phantom.

    Parameter
    --------
    space: `odl Discretized Space`
    p = 0.5: probability to choose sheep_logan.

    Returns
    --------
    phantom: odl phantom

    Example
    --------
    s = odl.uniform_discr([-1,-1],[1,1],[256,256],dtype='float32')
    p = gen_phantom(s)
    """

    c = rng.rand()
    if c<=p:
        p = rand_shep_logan_phantom(space)
    else:
        p = rand_forbild(space)
    return p

def rand_forbild(space, resolution=True, ear=True, value_type='density',
            scale='auto'):
    
    def transposeravel(arr):
        """Implement MATLAB's ``transpose(arr(:))``."""
        return arr.T.ravel()

    if not isinstance(space, DiscreteLp):
        raise TypeError('`space` must be a `DiscreteLp`')
    if space.ndim != 2:
        raise TypeError('`space` must be two-dimensional')

    # Create analytic description of phantom
    phantomE, phantomC = rand_forbild_phantom(resolution, ear)

    # Rescale points to the default grid.
    # The forbild phantom is defined on [-12.8, 12.8] x [-12.8, 12.8]
    xcoord, ycoord = space.points().T
    if scale == 'auto':
        xcoord = ((xcoord - space.min_pt[0]) /
                  (space.max_pt[0] - space.min_pt[0]))
        xcoord = 25.8 * xcoord - 12.8
        ycoord = ((ycoord - space.min_pt[1]) /
                  (space.max_pt[1] - space.min_pt[1]))
        ycoord = 25.8 * ycoord - 12.8
    elif scale == 'cm':
        pass  # dimensions already correct.
    elif scale == 'm':
        xcoord *= 100.0
        ycoord *= 100.0
    elif scale == 'mm':
        xcoord /= 10.0
        ycoord /= 10.0
    else:
        raise ValueError('unknown `scale` {}'.format(scale))

    # Compute the phantom values in each voxel
    image = np.zeros(space.size)
    nclipinfo = 0
    for k in range(phantomE.shape[0]):
        # Handle elliptic bounds
        Vx0 = np.array([transposeravel(xcoord) - phantomE[k, 0],
                        transposeravel(ycoord) - phantomE[k, 1]])
        D = np.array([[1 / phantomE[k, 2], 0],
                      [0, 1 / phantomE[k, 3]]])
        phi = np.deg2rad(phantomE[k, 4])
        Q = np.array([[np.cos(phi), np.sin(phi)],
                      [-np.sin(phi), np.cos(phi)]])
        f = phantomE[k, 5]
        nclip = int(phantomE[k, 6])
        equation1 = np.sum(D.dot(Q).dot(Vx0) ** 2, axis=0)
        i = (equation1 <= 1.0)

        # Handle clipping surfaces
        for _ in np.arange(nclip, dtype=object):  # note: nclib can be 0
            d = phantomC[0, nclipinfo]
            psi = np.deg2rad(phantomC[1, nclipinfo])
            equation2 = np.array([np.cos(psi), np.sin(psi)]).dot(Vx0)
            i &= (equation2 < d)
            nclipinfo += 1

        image[i] += f

    if value_type == 'materials':
        materials = np.zeros(space.size, dtype=space.dtype)
        # csf
        materials[(image > 1.043) & (image <= 1.047)] = 1
        # less_dense_sphere
        materials[(image > 1.047) & (image <= 1.048)] = 2
        # brain
        materials[(image > 1.048) & (image <= 1.052)] = 3
        # denser_sphere
        materials[(image > 1.052) & (image <= 1.053)] = 4
        # blood
        materials[(image > 1.053) & (image <= 1.058)] = 5
        # eye
        materials[(image > 1.058) & (image <= 1.062)] = 6
        # Bone
        materials[image > 1.75] = 7

        return space.element(materials)
    elif value_type == 'density':
        return space.element((image/np.abs(image).max()).reshape((space.shape[0],space.shape[1])))
    else:
        raise ValueError('unknown `value_type` {}'.format(value_type))

def rand_shep_logan_phantom(space):

    ellipses = rand_sheep_logan_ellipse()

    # Blank image
    p = np.zeros(space.shape, dtype=space.dtype)

    # Create the pixel grid
    grid_in = space.grid.meshgrid
    minp = space.grid.min()
    maxp = space.grid.max()

    # move points to [-1, 1]
    grid = []
    for i in range(2):
        meani = (minp[i] + maxp[i]) / 2.0
        # Where space.shape = 1, we have minp = maxp, so we set diffi = 1
        # to avoid division by zero. Effectively, this allows constructing
        # a slice of a 2D phantom.
        diffi = (maxp[i] - minp[i]) / 2.0 or 1.0
        grid += [(grid_in[i] - meani) / diffi]

    for ellip in ellipses:

        intensity = ellip[0]
        a_squared = ellip[1] ** 2
        b_squared = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        theta = ellip[5] * np.pi / 180

        scales = [1 / a_squared, 1 / b_squared]
        center = (np.array([x0, y0]) + 1.0) / 2.0

        # Create the offset x,y and z values for the grid
        if theta != 0:
            # Rotate the points to the expected coordinate system.
            ctheta = np.cos(theta)
            stheta = np.sin(theta)

            mat = np.array([[ctheta, stheta],
                            [-stheta, ctheta]])

            # Calculate the points that could possibly be inside the volume
            # Since the points are rotated, we cannot do anything directional
            # without more logic
            max_radius = np.sqrt(
                np.abs(mat).dot([a_squared, b_squared]))
            idx, shapes = _getshapes_2d(center, max_radius, space.shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            offset_points = [vec * (xi - x0i)[..., np.newaxis]
                             for xi, vec, x0i in zip(subgrid,
                                                     mat.T,
                                                     [x0, y0])]
            rotated = offset_points[0] + offset_points[1]
            np.square(rotated, out=rotated)
            radius = np.dot(rotated, scales)
        else:
            # Calculate the points that could possibly be inside the volume
            max_radius = np.sqrt([a_squared, b_squared])
            idx, shapes = _getshapes_2d(center, max_radius, space.shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            squared_dist = [ai * (xi - x0i) ** 2
                            for xi, ai, x0i in zip(subgrid,
                                                   scales,
                                                   [x0, y0])]

            # Parentheses to get best order for broadcasting
            radius = squared_dist[0] + squared_dist[1]

        # Find the pixels within the ellipse
        inside = radius <= 1

        # Add the ellipse intensity to those pixels
        p[idx][inside] += intensity

        # add random cuboids
        r1 = [rng.rand(20) - 0.5, rng.rand(20) - 0.5]
        r2 = [r1[0]+0.2*(rng.rand(20)-0.5), r1[1]+0.2*(rng.rand(20)-0.5)]
        i = 0
        n = rng.randint(0,3)
        sc = rng.randn(n)*0.6 + 0.4
        while i<n:
            p += np.multiply(sc[i], cuboid(space, [r1[0][i],r1[1][i]], [r2[0][i],r2[1][i]]).asarray())
            i += 1

    return space.element(p/np.abs(p).max())

def rand_forbild_phantom(resolution, ear):
    sha = 0.2 * np.sqrt(3)
    y016b = -14.294530834372887
    a16b = 0.443194085308632
    b16b = 3.892760834372886

    E = [[-4.7, 4.3, 1.79989, 1.79989, 0, 0.010, 0],  # 1
            [4.7, 4.3, 1.79989, 1.79989, 0, 0.010, 0],  # 2
            [-1.08, -9, 0.4, 0.4, 0, 0.0025, 0],  # 3
            [1.08, -9, 0.4, 0.4, 0, -0.0025, 0],  # 4
            [0, 0, 9.6, 12, 0, 1.800, 0],  # 5
            [0, 8.4, 1.8, 3.0, 0, -1.050, 0],  # 7
            [1.9, 5.4, 0.41633, 1.17425, -31.07698, 0.750, 0],  # 8
            [-1.9, 5.4, 0.41633, 1.17425, 31.07698, 0.750, 0],  # 9
            [-4.3, 6.8, 1.8, 0.24, -30, 0.750, 0],  # 10
            [4.3, 6.8, 1.8, 0.24, 30, 0.750, 0],  # 11
            [0, -3.6, 1.8, 3.6, 0, -0.005, 0],  # 12
            [6.39395, -6.39395, 1.2, 0.42, 58.1, 0.005, 0],  # 13
            [0, 3.6, 2, 2, 0, 0.750, 4],  # 14
            [0, 9.6, 1.8, 3.0, 0, 1.800, 4],  # 15
            [0, 0, 9.0, 11.4, 0, 0.750, 3],  # 16a
            [0, y016b, a16b, b16b, 0, 0.750, 1],  # 16b
            [0, 0, 9.0, 11.4, 0, -0.750, ear],  # 6
            [9.1, 0, 4.2, 1.8, 0, 0.750, 1]]  # R_ear
    r = 5*(rng.rand(200)*0.2 - 0.1)
    c = 0
    for i in np.arange(0,18):
        for j in np.arange(0,6):
            E[i][j] += r[c]
            c = c + 1
    idx = rng.randint(2,13,3)
    E = np.array(E)

    # generate the air cavities in the right ear
    cavity1 = np.arange(8.8, 5.6, -0.4)[:, None]
    cavity2 = np.zeros([9, 1])
    cavity3_7 = np.ones([53, 1]) * [0.15, 0.15, 0, -1.800, 0]

    for j in np.arange(1, 4):
        kj = 8 - 2 * int(np.floor(j / 3))
        Ej = 0.2 * int(np.mod(j, 2))

        cavity1 = np.vstack((cavity1,
            cavity1[0:kj] - Ej,
            cavity1[0:kj] - Ej))
        cavity2 = np.vstack((cavity2,
            j * sha * np.ones([kj, 1]),
            -j * sha * np.ones([kj, 1])))

    E_cavity = np.hstack((cavity1, cavity2, cavity3_7))

    # generate the left ear (resolution pattern)
    x0 = -7.0
    y0 = -1.0
    E0_xy = 0.04

    E_xy = [0.0357, 0.0312, 0.0278, 0.0250]
    ab = 0.5 * np.ones([5, 1]) * E_xy
    ab = ab.T.ravel()[:, None] * np.ones([1, 4])
    abr = ab.T.ravel()[:, None]

    leftear4_7 = np.hstack([abr, abr, np.ones([80, 1]) * [0, 0.75, 0]])

    x00 = np.zeros([0, 1])
    y00 = np.zeros([0, 1])
    for i in np.arange(1, 5):
        y00 = np.vstack((y00,
            (y0 + np.arange(0, 5) * 2 * E_xy[i - 1])[:, None]))
        x00 = np.vstack((x00,
            (x0 + 2 * (i - 1) * E0_xy) * np.ones([5, 1])))

    x00 = x00 * np.ones([1, 4])
    x00 = x00.T.ravel()[:, None]
    y00 = np.vstack([y00, y00 + 12 * E0_xy,
        y00 + 24 * E0_xy, y00 + 36 * E0_xy])

    leftear = np.hstack([x00, y00, leftear4_7])
    C = [[1.2, 1.2, 0.27884, 0.27884, 0.60687, 0.60687, 0.2,
        0.2, -2.605, -2.605, -10.71177, y016b + 10.71177, 8.88740, -0.21260],
        [0, 180, 90, 270, 90, 270, 0,
            180, 15, 165, 90, 270, 0, 0]]
    C = np.array(C)

    if not resolution and not ear:
        phantomE = E[:17, :]
        phantomC = C[:, :12]
    elif not resolution and ear:
        phantomE = np.vstack([E, E_cavity])
        phantomC = C
    elif resolution and not ear:
        phantomE = np.vstack([leftear, E[:17, :]])
        phantomC = C[:, :12]
    else:
        phantomE = np.vstack([leftear, E, E_cavity])
        phantomC = C

    return phantomE, phantomC

def rand_sheep_logan_ellipse():
    rot = rng.randint(-90,90, size=10)
    rot[0] = rng.randint(-10,10)
    rot[1] = rng.randint(-10,10)
    d = [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.2, .1100, .3100, 0.2200, 0.0000, -18],
            [-.2, .1600, .4100, -.2200, 0.0000, 18],
            [0.1, .2100, .2500, 0.0000, 0.3500, 0],
            [0.1, .0460, .0460, 0.0000, 0.1000, 0],
            [0.1, .0460, .0460, 0.0000, -.1000, 0],
            [0.1, .0460, .0230, -.0800, -.6050, 0],
            [0.1, .0230, .0230, 0.0000, -.6060, 0],
            [0.1, .0230, .0460, 0.0600, -.6050, 0]]
    for i in range(0,10):
        d[i][5] = rot[i]
    r = (rng.rand(2)*0.3 - 0.15)
    d[0][0] += r[0]
    d[1][0] += r[0]
    d[0][1] += r[1]
    d[1][1] += r[1]
    r = 3*(rng.rand(20)*0.2 - 0.1)
    c = 0
    for i in np.arange(2,10):
        for j in np.arange(1,3):
            d[i][j] += r[c]
            c = c + 1
    c = 0
    r = rng.rand(18)-0.5
    for i in np.arange(2,10):
        for j in np.arange(3,5):
            d[i][j] += r[c]
            c = c + 1
    i = 0
    n = rng.randint(0, 5)
    h = [0.3, -0.3, 0.4 , -0.5]
    while i<n:
        l = [h[rng.randint(0,3)], rng.rand()*0.2, rng.rand()*4, rng.randn(), rng.randn(), rng.randint(-30, 30)]
        d.append(l)
        i += 1
    return d
