# -*- coding: utf-8 -*-
"""GLI-2012 spirometry reference values (Quanjer et al, ERJ 2012).

Port of the published LMS model to Python. The L/M/S spline lookup table is the
official GLI-2012 table as redistributed in the rspiro R package
(data-raw/RLookupTable.csv, https://github.com/thlytras/rspiro); the equation
below is the one rspiro's getLMS.R implements:

    Lspline = l0 + (l1-l0)*(age-agebound)/0.25          (same for M, S)
    L = q0 + q1*ln(age) + Lspline
    M = exp(a0 + a1*ln(height_cm) + a2*ln(age) + ethnicity_term + Mspline)
    S = exp(p0 + p1*ln(age) + ethnicity_term + Sspline)
    z = ((y/M)^L - 1) / (L*S)
    LLN = M * (1 - 1.645*L*S)^(1/L)      (5th percentile)

Ethnicity codes: 1 Caucasian, 2 African-American, 3 NE Asian, 4 SE Asian, 5 Other.
"""
import csv, math, os

_TABLE = None
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RLookupTable.csv')


def _load():
    global _TABLE
    if _TABLE is None:
        _TABLE = {}
        for r in csv.DictReader(open(_PATH)):
            key = (int(r['sex']), r['f'], round(float(r['agebound']), 2))
            _TABLE[key] = {k: (float(v) if v not in ('', 'NA') else 0.0)
                           for k, v in r.items() if k not in ('sex', 'f')}
    return _TABLE


def lms(age, height_cm, sex, param, ethnicity=3):
    """L, M, S for one subject. sex: 1 male, 2 female. age in years, height in cm."""
    t = _load()
    agebound = math.floor(age * 4) / 4
    row = t.get((sex, param, round(agebound, 2)))
    if row is None:
        raise ValueError('age %s out of GLI-2012 range for %s' % (age, param))
    frac = (age - agebound) / 0.25
    Lsp = row['l0'] + (row['l1'] - row['l0']) * frac
    Msp = row['m0'] + (row['m1'] - row['m0']) * frac
    Ssp = row['s0'] + (row['s1'] - row['s0']) * frac
    eth_m = {2: row['a3'], 3: row['a4'], 4: row['a5'], 5: row['a6']}.get(ethnicity, 0.0)
    eth_s = {2: row['p2'], 3: row['p3'], 4: row['p4'], 5: row['p5']}.get(ethnicity, 0.0)
    L = row['q0'] + row['q1'] * math.log(age) + Lsp
    M = math.exp(row['a0'] + row['a1'] * math.log(height_cm) + row['a2'] * math.log(age)
                 + eth_m + Msp)
    S = math.exp(row['p0'] + row['p1'] * math.log(age) + eth_s + Ssp)
    return L, M, S


def zscore(value, age, height_cm, sex, param, ethnicity=3):
    L, M, S = lms(age, height_cm, sex, param, ethnicity)
    return ((value / M) ** L - 1) / (L * S)


def lln(age, height_cm, sex, param, ethnicity=3):
    L, M, S = lms(age, height_cm, sex, param, ethnicity)
    return M * (1 - 1.645 * L * S) ** (1 / L)


def predicted(age, height_cm, sex, param, ethnicity=3):
    return lms(age, height_cm, sex, param, ethnicity)[1]


if __name__ == '__main__':
    # self-consistency checks
    worst_lln, worst_m, worst_end = 0.0, 0.0, 0.0
    t = _load()
    for sex in (1, 2):
        for param in ('FEV1', 'FVC', 'FEV1FVC'):
            for age in (30.0, 45.5, 60.13, 72.25, 88.9):
                for h in (145.0, 165.0, 185.0):
                    L, M, S = lms(age, h, sex, param)
                    worst_lln = max(worst_lln, abs(
                        zscore(lln(age, h, sex, param), age, h, sex, param) + 1.645))
                    worst_m = max(worst_m, abs(zscore(M, age, h, sex, param)))
            # interpolation must reproduce the table exactly at an agebound
            for ab in (40.0, 65.25, 80.5):
                row = t[(sex, param, ab)]
                L, M, S = lms(ab, 170.0, sex, param)
                Lref = row['q0'] + row['q1'] * math.log(ab) + row['l0']
                worst_end = max(worst_end, abs(L - Lref))
    print('z(LLN) vs -1.645 : max err %.2e' % worst_lln)
    print('z(M)    vs 0     : max err %.2e' % worst_m)
    print('spline at agebound: max err %.2e' % worst_end)
    # a couple of concrete cases the user can re-check on gli-calculator.ersnet.org
    for age, h, sex, lab in ((65.0, 170.0, 1, 'male 65y 170cm'),
                             (70.0, 155.0, 2, 'female 70y 155cm')):
        print('%s, NE Asian:' % lab,
              'FEV1 pred %.3f L, LLN %.3f' % (predicted(age, h, sex, 'FEV1'),
                                              lln(age, h, sex, 'FEV1')),
              '| FVC pred %.3f L, LLN %.3f' % (predicted(age, h, sex, 'FVC'),
                                               lln(age, h, sex, 'FVC')),
              '| FEV1/FVC pred %.1f%%, LLN %.1f%%' % (predicted(age, h, sex, 'FEV1FVC') * 100,
                                                      lln(age, h, sex, 'FEV1FVC') * 100))
