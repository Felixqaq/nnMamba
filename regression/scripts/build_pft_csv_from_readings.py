"""Write the 61 newly-read PFT rows in the same shape as fev1_fvc.csv.

FEV1FVC_pct follows the existing convention: post-bronchodilator where the report
has those columns, otherwise pre. Obstruction_fixed70 is the GOLD fixed ratio,
strictly < 70. Check_calc recomputes the ratio from the printed volumes so a
misread stands out; the largest disagreement across all 61 pages is 0.69 points.

FVL_ECode is the analyser's quality flag on the flow-volume manoeuvre, carried
through because a non-zero code means the effort was not clean — the same reason
two patients were dropped from the current cohort.
"""
import csv, io, importlib.util as u

spec = u.spec_from_file_location("r", "pft_new_readings.py")
m = u.module_from_spec(spec); spec.loader.exec_module(m)

OUT = "fev1_fvc_new_batches.csv"
fields = ["Date","PatientID","FEV1FVC_pct","Source","FEV1FVC_pre","FEV1FVC_post",
          "FEV1FVC_ref","FVC_pre","FEV1_pre","FVC_post","FEV1_post","FVC_ref","FEV1_ref",
          "Age","Sex","Height_cm","Weight_kg","FVL_ECode_pre","FVL_ECode_post",
          "Obstruction_fixed70","Check_calc","Check_diff"]

rows = []
for (d,pid,sex,age,h,w, fr,er,rr, fp,ep,rp, fo,eo,ro, c1,c2) in m.D:
    use_post = ro is not None
    ratio = ro if use_post else rp
    fvc, fev1 = (fo, eo) if use_post else (fp, ep)
    calc = round(fev1 / fvc * 100, 1)
    rows.append({
        "Date": d, "PatientID": pid,
        "FEV1FVC_pct": ratio, "Source": "Post" if use_post else "Pre",
        "FEV1FVC_pre": rp, "FEV1FVC_post": ro if ro is not None else "",
        "FEV1FVC_ref": rr,
        "FVC_pre": fp, "FEV1_pre": ep,
        "FVC_post": fo if fo is not None else "", "FEV1_post": eo if eo is not None else "",
        "FVC_ref": fr, "FEV1_ref": er,
        "Age": age, "Sex": sex, "Height_cm": h, "Weight_kg": w,
        "FVL_ECode_pre": c1 or "", "FVL_ECode_post": c2 or "",
        "Obstruction_fixed70": "Y" if ratio < 70 else "N",
        "Check_calc": calc, "Check_diff": round(calc - ratio, 1),
    })

with io.open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

import collections
obs = collections.Counter(r["Obstruction_fixed70"] for r in rows)
src = collections.Counter(r["Source"] for r in rows)
flagged = [r["PatientID"] for r in rows
           if (r["FVL_ECode_pre"] and r["FVL_ECode_pre"] != "000000")
           or (r["FVL_ECode_post"] and r["FVL_ECode_post"] != "000000")]
print(f"wrote {OUT}: {len(rows)} rows")
print(f"  Obstruction_fixed70: {dict(obs)}  ->  Abnormal {obs['Y']} / Normal {obs['N']}")
print(f"  Source: {dict(src)}")
print(f"  worst |Check_diff|: {max(abs(r['Check_diff']) for r in rows)}")
print(f"  non-zero FVL ECode: {len(flagged)} {flagged}")
near = [(r["PatientID"], r["FEV1FVC_pct"]) for r in rows if abs(r["FEV1FVC_pct"] - 70) < 5]
print(f"  within 5 points of the 70 cut: {len(near)} {near}")
