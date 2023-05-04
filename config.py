import configparser
import numpy as np
from precision import *


def getParameters(conf_str, type=cst):
    return np.array([type(val) for val in conf_str.split(',')])


def readConfig(parafile):
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(parafile)

    # Parameters
    γρ = getParameters(config['Parameters']['γρ'])
    g = getParameters(config['Parameters']['g'])
    ν = getParameters(config['Parameters']['ν'])
    μ = getParameters(config['Parameters']['μ'])
    ## Additional parameters
    try:
        labels = getParameters(config['Parameters']['label'], str)
    except KeyError:
        labels = None
    colors = ['blue', 'green', 'red', 'orange', 'yellow']

    # Discretization
    Nx = int(config['Discretization']['Nx'])
    Nt = int(config['Discretization']['Nt'])
    dt = cst(config['Discretization']['dt'])
    periodic = bool(config['Discretization']['periodic'] == 'True')
    filter = bool(config['Discretization']['filter'] == 'True')
    try:
        normalized = bool(config['Discretization']['normalized'] == 'True')
    except KeyError:
        normalized = True

    # Domain
    try:
        x, dx = np.linspace(0., cst(config['Parameters']['xf']), Nx, dtype=wp, retstep=True, endpoint=not periodic)
    except KeyError:
        x, dx = np.linspace(0., 1., Nx, dtype=wp, retstep=True, endpoint=not periodic)
    dx = cst(dx)  # stupid convention

    # Initial
    v0 = np.vectorize(lambda z: eval(config['Initial']['v0']))
    h0 = np.vectorize(lambda z: eval(config['Initial']['h0']))
    # Reference
    try:
        _ = config['Initial']['vr']
        vr = np.vectorize(lambda t, z, v0=v0, h0=h0: eval(config['Initial']['vr']))
    except KeyError:
        vr = None
    try:
        _ = config['Initial']['hr']
        hr = np.vectorize(lambda t, z, v0=v0, h0=h0: eval(config['Initial']['hr']))
    except KeyError:
        hr = None

    # Adjust size for all
    noruns = np.max([len(val) for val in [γρ, g, ν, μ]])
    γρ = np.repeat(γρ, noruns) if len(γρ) != noruns else γρ
    g  = np.repeat(g, noruns)  if len(g)  != noruns else g
    ν  = np.repeat(ν, noruns)  if len(ν)  != noruns else ν
    μ  = np.repeat(μ, noruns)  if len(μ)  != noruns else μ

    return (γρ, g, ν, μ), (x, Nx, dx, Nt, dt, periodic, filter, normalized), (v0, h0, vr, hr), (labels, colors)