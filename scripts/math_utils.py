import sys 
sys.path.append(".")

import math

def inch2meter(value_inch):
    return value_inch * 0.0254

def meter2inch(value_meter):
    return value_meter / 0.0254

def calc_avg_sphere_volume(min_d, max_d):
    """calculate expected volume in range [min_d, max_d]"""
    min_r = min_d / 2
    max_r = max_d / 2
    # f = 4/3 * math.pi * r^3
    # F = 1/3 * math.pi * r^4
    F = lambda r: 1/3 * math.pi * r**4
    
    return (F(max_r) - F(min_r)) / (max_r - min_r)


def pdf2cdf(pdf):
    """transfer probability densiity function to cummulative distribution function"""
    total = sum(pdf)
    cdf = [0]

    s = 0
    
    for p in pdf[:-1]:
        s += p 
        cdf.append(s / total)
    return cdf

def cdf2pdf(cdf):
    """transfer cummulative distribution function to probability density function"""
    pdf = []
    for i in range(len(cdf) - 1):
        pdf.append(cdf[i+1] - cdf[i])
    pdf.append(0)
    return pdf

def calc_sphere_vol(diameter):
    return 4/3*math.pi*(diameter/2)**3

def calc_sphere_diameter_from_vol(vol):
    return 2 * (vol/4/math.pi*3) ** (1/3)