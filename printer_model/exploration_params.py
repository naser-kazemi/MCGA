# NOTE: this is a sample params file for testing purpose only
import numpy as np
from math import pi

# color exploration
#
# objectives:
# - hue maximization
# - ds diversity maximization
# - ps crowdness maximization
# - hue wheel
# - thickness minimization

# marking
# model = "nsga2"
model = "mc_nsga2"
# model = "mc_nsga3"
# model = "nsga3"
name = "2.0_0.2"
marking_area = np.array([120, 120])
calib_params = [227265, 600, 75.0, 40, 1]
calib_offs = -1.2

rep_patches_params = np.array([])

avg_lt = 0.05
mark_calibration = 1

cross_dens = np.array([7, 7])
patch_index_off = np.array([1, 1])
patch_align = np.array([7, 7])

patch_width = 1
patch_dist = 0.1
corner_coords = [0, 0]
offs_per_idx = np.array([15, 15])

random_start = 1

# genetic algorithm
hw = 1
objs = np.array([1, 1, 1, 0, 0])  # c, dsd, psd, t, l*
pop_size = 100
iterations = 30
cross_prob = 0.8
mut_prob = 0.2
k_n = 2
energy_lims = [0.01, 200]
start_angle_range = [0, 2 * pi]
cone_count_range = [4, 72]
hyp_lt = 0.001
bits_per_param = [10, 10, 10]
prec_facs = [1000, 1000, 1000]
min_ppl = 40
max_t = 0.08
cont_ds = [1, 1, 1]
limits_ps = np.array([[0, 100], [-100, 100], [-100, 100]], dtype=np.float32)
limits_ps = [np.array(arr, dtype=np.float32) for arr in limits_ps]
limits_ds = [[0, 1], [0, 1], [0, 1]]

limits_ds = [np.array(arr, dtype=np.float32) for arr in limits_ds]

xyz2rgb = np.array(
    [
        [3.240454200000000, -1.537138500000000, -0.498531400000000],
        [-0.969266000000000, 1.876010800000000, 0.041556000000000],
        [0.055643400000000, -0.204025900000000, 1.057225200000000],
    ],
    dtype=np.float32,
)

d65_illuminant = np.array(
    [
        82.7500000000000,
        91.4900000000000,
        93.4300000000000,
        86.6800000000000,
        104.860000000000,
        117.010000000000,
        117.810000000000,
        114.860000000000,
        115.920000000000,
        108.810000000000,
        109.350000000000,
        107.800000000000,
        104.790000000000,
        107.690000000000,
        104.410000000000,
        104.050000000000,
        100,
        96.3300000000000,
        95.7900000000000,
        88.6900000000000,
        90.0100000000000,
        89.6000000000000,
        87.7000000000000,
        83.2900000000000,
        83.7000000000000,
        80.0300000000000,
        80.2100000000000,
        82.2800000000000,
        78.2800000000000,
        69.7200000000000,
        71.6100000000000,
    ]
)
xbar = np.array(
    [
        0.0143000000000000,
        0.0435000000000000,
        0.134400000000000,
        0.283900000000000,
        0.348300000000000,
        0.336200000000000,
        0.290800000000000,
        0.195400000000000,
        0.0956000000000000,
        0.0320000000000000,
        0.00490000000000000,
        0.00930000000000000,
        0.0633000000000000,
        0.165500000000000,
        0.290400000000000,
        0.433400000000000,
        0.594500000000000,
        0.762100000000000,
        0.916300000000000,
        1.02630000000000,
        1.06220000000000,
        1.00260000000000,
        0.854400000000000,
        0.642400000000000,
        0.447900000000000,
        0.283500000000000,
        0.164900000000000,
        0.0874000000000000,
        0.0468000000000000,
        0.0227000000000000,
        0.0114000000000000,
    ]
)
ybar = np.array(
    [
        0.000400000000000000,
        0.00120000000000000,
        0.00400000000000000,
        0.0116000000000000,
        0.0230000000000000,
        0.0380000000000000,
        0.0600000000000000,
        0.0910000000000000,
        0.139000000000000,
        0.208000000000000,
        0.323000000000000,
        0.503000000000000,
        0.710000000000000,
        0.862000000000000,
        0.954000000000000,
        0.995000000000000,
        0.995000000000000,
        0.952000000000000,
        0.870000000000000,
        0.757000000000000,
        0.631000000000000,
        0.503000000000000,
        0.381000000000000,
        0.265000000000000,
        0.175000000000000,
        0.107000000000000,
        0.0610000000000000,
        0.0320000000000000,
        0.0170000000000000,
        0.00820000000000000,
        0.00410000000000000,
    ]
)
zbar = np.array(
    [
        0.0679000000000000,
        0.207400000000000,
        0.645600000000000,
        1.38560000000000,
        1.74710000000000,
        1.77210000000000,
        1.66920000000000,
        1.28760000000000,
        0.813000000000000,
        0.465200000000000,
        0.272000000000000,
        0.158200000000000,
        0.0782000000000000,
        0.0422000000000000,
        0.0203000000000000,
        0.00870000000000000,
        0.00390000000000000,
        0.00210000000000000,
        0.00170000000000000,
        0.00110000000000000,
        0.000800000000000000,
        0.000300000000000000,
        0.000200000000000000,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
)
spectra_start_wl = 400
spectra_num_wl = 31
spectra_specular = 0

white_ciexyz = np.array([94.9399572601785, 100.000000000000, 108.706164596135])

ng_primary_reflectances = np.array(
    [
        [
            0.413700000000000,
            0.780900000000000,
            1.07930000000000,
            1.16120000000000,
            1.17110000000000,
            1.09720000000000,
            1.01770000000000,
            0.981900000000000,
            0.956900000000000,
            0.928300000000000,
            0.915200000000000,
            0.902500000000000,
            0.890400000000000,
            0.880800000000000,
            0.875600000000000,
            0.872400000000000,
            0.868500000000000,
            0.865700000000000,
            0.862600000000000,
            0.857800000000000,
            0.852400000000000,
            0.849900000000000,
            0.847600000000000,
            0.846300000000000,
            0.845800000000000,
            0.845300000000000,
            0.846600000000000,
            0.849200000000000,
            0.851500000000000,
            0.852200000000000,
            0.853200000000000,
        ],
        [
            0.188000000000000,
            0.275600000000000,
            0.393100000000000,
            0.583200000000000,
            0.797000000000000,
            0.890100000000000,
            0.888200000000000,
            0.870000000000000,
            0.842000000000000,
            0.801400000000000,
            0.755500000000000,
            0.694600000000000,
            0.617200000000000,
            0.526900000000000,
            0.432500000000000,
            0.346300000000000,
            0.270800000000000,
            0.208100000000000,
            0.163000000000000,
            0.136100000000000,
            0.122300000000000,
            0.113000000000000,
            0.104200000000000,
            0.0992000000000000,
            0.0988000000000000,
            0.104000000000000,
            0.120900000000000,
            0.166900000000000,
            0.276500000000000,
            0.447000000000000,
            0.621800000000000,
        ],
        [
            0.334300000000000,
            0.590600000000000,
            0.784500000000000,
            0.811100000000000,
            0.775100000000000,
            0.679800000000000,
            0.586200000000000,
            0.513400000000000,
            0.439500000000000,
            0.381400000000000,
            0.340100000000000,
            0.283100000000000,
            0.234100000000000,
            0.224000000000000,
            0.231800000000000,
            0.222400000000000,
            0.229000000000000,
            0.304000000000000,
            0.454700000000000,
            0.621400000000000,
            0.736100000000000,
            0.792400000000000,
            0.816300000000000,
            0.826100000000000,
            0.829500000000000,
            0.831000000000000,
            0.833400000000000,
            0.837300000000000,
            0.840400000000000,
            0.842200000000000,
            0.843200000000000,
        ],
        [
            0.186700000000000,
            0.291300000000000,
            0.361300000000000,
            0.380600000000000,
            0.394200000000000,
            0.390300000000000,
            0.392800000000000,
            0.420300000000000,
            0.464600000000000,
            0.523900000000000,
            0.606200000000000,
            0.691100000000000,
            0.757700000000000,
            0.800300000000000,
            0.826400000000000,
            0.841500000000000,
            0.849000000000000,
            0.852500000000000,
            0.852800000000000,
            0.849500000000000,
            0.845200000000000,
            0.843400000000000,
            0.841800000000000,
            0.841500000000000,
            0.840500000000000,
            0.839900000000000,
            0.841200000000000,
            0.844000000000000,
            0.846300000000000,
            0.846800000000000,
            0.847500000000000,
        ],
        [
            0.152400000000000,
            0.218800000000000,
            0.260400000000000,
            0.267000000000000,
            0.266900000000000,
            0.255700000000000,
            0.247300000000000,
            0.247600000000000,
            0.247000000000000,
            0.251000000000000,
            0.257600000000000,
            0.239800000000000,
            0.213600000000000,
            0.213000000000000,
            0.225200000000000,
            0.218600000000000,
            0.226600000000000,
            0.302600000000000,
            0.455000000000000,
            0.622800000000000,
            0.737700000000000,
            0.794300000000000,
            0.818300000000000,
            0.828300000000000,
            0.832100000000000,
            0.833500000000000,
            0.836200000000000,
            0.839900000000000,
            0.842800000000000,
            0.843900000000000,
            0.845300000000000,
        ],
        [
            0.0921000000000000,
            0.117300000000000,
            0.148800000000000,
            0.195200000000000,
            0.246200000000000,
            0.277100000000000,
            0.300100000000000,
            0.334600000000000,
            0.381300000000000,
            0.438400000000000,
            0.500500000000000,
            0.538400000000000,
            0.532200000000000,
            0.484900000000000,
            0.412700000000000,
            0.335600000000000,
            0.262600000000000,
            0.199000000000000,
            0.151400000000000,
            0.121600000000000,
            0.105300000000000,
            0.0938000000000000,
            0.0828000000000000,
            0.0762000000000000,
            0.0755000000000000,
            0.0824000000000000,
            0.103100000000000,
            0.153500000000000,
            0.265300000000000,
            0.437900000000000,
            0.615800000000000,
        ],
        [
            0.155600000000000,
            0.210800000000000,
            0.293600000000000,
            0.420100000000000,
            0.533300000000000,
            0.549400000000000,
            0.507600000000000,
            0.452400000000000,
            0.387000000000000,
            0.333800000000000,
            0.290100000000000,
            0.232300000000000,
            0.181200000000000,
            0.156900000000000,
            0.141500000000000,
            0.116100000000000,
            0.0970000000000000,
            0.0941000000000000,
            0.0967000000000000,
            0.0960000000000000,
            0.0918000000000000,
            0.0844000000000000,
            0.0755000000000000,
            0.0695000000000000,
            0.0697000000000000,
            0.0782000000000000,
            0.100900000000000,
            0.153300000000000,
            0.266000000000000,
            0.437600000000000,
            0.613300000000000,
        ],
        [
            0.0786000000000000,
            0.0896000000000000,
            0.109100000000000,
            0.140800000000000,
            0.171300000000000,
            0.185900000000000,
            0.194400000000000,
            0.203600000000000,
            0.210100000000000,
            0.218200000000000,
            0.222600000000000,
            0.201800000000000,
            0.170700000000000,
            0.155200000000000,
            0.144200000000000,
            0.121200000000000,
            0.102100000000000,
            0.0960000000000000,
            0.0920000000000000,
            0.0840000000000000,
            0.0750000000000000,
            0.0647000000000000,
            0.0541000000000000,
            0.0476000000000000,
            0.0474000000000000,
            0.0556000000000000,
            0.0790000000000000,
            0.136000000000000,
            0.257400000000000,
            0.436300000000000,
            0.615400000000000,
        ],
    ]
)

num_divisions = 8
eta_crossover = 20
eta_mutation = 20
crossover_probability = 0.8
polar_offset_limit = [0, 2 * pi]
num_max_sectors = [10, 50]
front_frequency_threshold = 0.01
monte_carlo_frequency = 5
log = ["hv"]
verbose = True
nd = "log"
