# -*- coding: utf-8 -*-
"""GLI Global (2022) race-neutral spirometry reference values.

Bowerman et al, AJRCCM 2023 — the equations ERS/ATS 2022 points to. Coefficients
and the M/S spline lookup are the official GLI global tables
(gli_global_lookuptables_dec6.xlsx) as redistributed in rspiro; the equations
match rspiro's getLMS_GLIgl.R. L is a constant (FEV1, FVC) or a function of
ln(age) (FEV1/FVC), and the published implementation does not interpolate the
splines between agebounds -- it uses the value at floor(age*4)/4.

    M = exp(a0 + a1*ln(height_cm) + a2*ln(age) + Mspline)
    S = exp(p0 + p1*ln(age) + Sspline)
    z = ((y/M)^L - 1) / (L*S)
    LLN = M * (1 - 1.645*L*S)^(1/L)
"""
import math, os, csv

_DIR = os.path.dirname(os.path.abspath(__file__))
_XLSX = os.path.join(_DIR, 'gli_global.xlsx')
_CSV = os.path.join(_DIR, 'gli_global_splines.csv')

# (sex, param) -> (a0, a1, a2, p0, p1, L or None)
COEF = {
    (1, 'FEV1'):    (-11.399108, 2.462664, -0.011394, -2.256278, 0.080729, 1.22703),
    (1, 'FVC'):     (-12.629131, 2.727421,  0.009174, -2.195595, 0.068466, 0.9346),
    (1, 'FEV1FVC'): (  1.022608, -0.218592, -0.027586, -2.882025, 0.068889, None),
    (2, 'FEV1'):    (-10.901689, 2.385928, -0.076386, -2.364047, 0.129402, 1.21388),
    (2, 'FVC'):     (-12.055901, 2.621579, -0.035975, -2.310148, 0.120428, 0.899),
    (2, 'FEV1FVC'): ( 0.9189568, -0.1840671, -0.0461306, -3.171582, 0.144358, None),
}
_L_RATIO = {1: (3.8243, -0.3328), 2: (6.6490, -0.9920)}   # L = c0 + c1*ln(age)

_SPL = None
_SHEETS = [(1, 'FEV1', 'Male FEV1'), (1, 'FVC', 'Male FVC'), (1, 'FEV1FVC', 'Male FEV1 FVC'),
           (2, 'FEV1', 'Female FEV1'), (2, 'FVC', 'Female FVC'), (2, 'FEV1FVC', 'Female FEV1 FVC')]


def _load():
    """Splines, cached to a plain CSV so the table is inspectable and xlsx-free."""
    global _SPL
    if _SPL is not None:
        return _SPL
    if not os.path.exists(_CSV):
        import openpyxl
        wb = openpyxl.load_workbook(_XLSX, data_only=True)
        with open(_CSV, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['sex', 'param', 'agebound', 'Mspline', 'Sspline'])
            for sex, param, sheet in _SHEETS:
                ws = wb[sheet]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0] is None or not isinstance(row[0], (int, float)):
                        continue
                    w.writerow([sex, param, round(float(row[0]), 2),
                                float(row[1]), float(row[2])])
    _SPL = {}
    for r in csv.DictReader(open(_CSV, encoding='utf-8')):
        _SPL[(int(r['sex']), r['param'], round(float(r['agebound']), 2))] = (
            float(r['Mspline']), float(r['Sspline']))
    return _SPL


def lms(age, height_cm, sex, param):
    spl = _load()
    agebound = round(math.floor(age * 4) / 4, 2)
    key = (sex, param, agebound)
    if key not in spl:
        raise ValueError('age %s out of GLI global range for %s' % (age, param))
    Msp, Ssp = spl[key]
    a0, a1, a2, p0, p1, Lc = COEF[(sex, param)]
    M = math.exp(a0 + a1 * math.log(height_cm) + a2 * math.log(age) + Msp)
    S = math.exp(p0 + p1 * math.log(age) + Ssp)
    if Lc is None:
        c0, c1 = _L_RATIO[sex]
        L = c0 + c1 * math.log(age)
    else:
        L = Lc
    return L, M, S


def zscore(value, age, height_cm, sex, param):
    L, M, S = lms(age, height_cm, sex, param)
    return ((value / M) ** L - 1) / (L * S)


def lln(age, height_cm, sex, param):
    L, M, S = lms(age, height_cm, sex, param)
    return M * (1 - 1.645 * L * S) ** (1 / L)


def predicted(age, height_cm, sex, param):
    return lms(age, height_cm, sex, param)[1]


if __name__ == '__main__':
    spl = _load()
    print('spline rows:', len(spl))
    wl = wm = 0.0
    for sex in (1, 2):
        for param in ('FEV1', 'FVC', 'FEV1FVC'):
            for age in (30.0, 45.5, 60.25, 72.0, 88.75):
                for h in (145.0, 165.0, 185.0):
                    wl = max(wl, abs(zscore(lln(age, h, sex, param), age, h, sex, param) + 1.645))
                    wm = max(wm, abs(zscore(lms(age, h, sex, param)[1], age, h, sex, param)))
    print('z(LLN) vs -1.645 : max err %.2e' % wl)
    print('z(M)    vs 0     : max err %.2e' % wm)
    for age, h, sex, lab in ((65.0, 170.0, 1, 'male 65y 170cm'),
                             (70.0, 155.0, 2, 'female 70y 155cm')):
        print('%s, GLI global:' % lab,
              'FEV1 pred %.3f LLN %.3f' % (predicted(age, h, sex, 'FEV1'), lln(age, h, sex, 'FEV1')),
              '| FVC pred %.3f LLN %.3f' % (predicted(age, h, sex, 'FVC'), lln(age, h, sex, 'FVC')),
              '| FEV1/FVC pred %.1f%% LLN %.1f%%' % (predicted(age, h, sex, 'FEV1FVC') * 100,
                                                     lln(age, h, sex, 'FEV1FVC') * 100))
