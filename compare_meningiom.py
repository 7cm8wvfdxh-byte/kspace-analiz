"""
Meningiom Bolgesi K-Space Karsilastirmasi
Frontal meningiom: Sag T1+C, Slice 9-13 bolgesi
"""
import csv

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# --- Sag T1+C (Pathological - with contrast, shows meningiom) ---
print("=" * 70)
print("  MENINGIOM K-SPACE ANALIZI - Sag T1+C")
print("=" * 70)

rad = read_csv('results_pathological_sag_t1c/kspace_radiomics_data.csv')
phase = read_csv('results_pathological_sag_t1c/kspace_phase_data.csv')

print("\n  RADIOMICS:")
print(f"  {'Slice':>5}  {'Entropy':>8}  {'LR_Asym':>8}  {'Skewness':>9}  {'Kurtosis':>9}  {'Note'}")
print("  " + "-" * 60)
for r in rad:
    s = int(r['slice'])
    note = "<-- MENINGIOM BOLGESI" if 9 <= s <= 13 else ""
    print(f"  {s:>5}  {float(r['entropy']):>8.2f}  {float(r['lr_asymmetry']):>8.4f}  {float(r['skewness']):>9.3f}  {float(r['kurtosis']):>9.1f}  {note}")

print("\n  PHASE COHERENCE:")
print(f"  {'Slice':>5}  {'PhaseGrad':>9}  {'Hermitian':>9}  {'Coherence':>9}  {'Note'}")
print("  " + "-" * 50)
for p in phase:
    s = int(p['slice'])
    note = "<-- MENINGIOM BOLGESI" if 9 <= s <= 13 else ""
    print(f"  {s:>5}  {float(p['phase_gradient_mean']):>9.4f}  {float(p['hermitian_error_mean']):>9.4f}  {float(p['mean_coherence']):>9.4f}  {note}")

# --- Compare with Normal Sag T1 ---
print("\n" + "=" * 70)
print("  KARSILASTIRMA: Normal Sag T1 vs Pathological Sag T1")
print("=" * 70)

norm_rad = read_csv('results_normal/kspace_radiomics_data.csv')
norm_phase = read_csv('results_normal/kspace_phase_data.csv')
path_rad = read_csv('results_pathological/kspace_radiomics_data.csv')
path_phase = read_csv('results_pathological/kspace_phase_data.csv')

# Compute averages
def avg(data, key):
    vals = [float(r[key]) for r in data]
    return sum(vals)/len(vals), min(vals), max(vals)

metrics = ['entropy', 'lr_asymmetry', 'skewness', 'kurtosis']
phase_metrics = ['phase_gradient_mean', 'hermitian_error_mean', 'mean_coherence']

print("\n  RADIOMICS OZET:")
print(f"  {'Metrik':<18}  {'Normal (Ort)':>14}  {'Normal (Min-Max)':>18}  {'Patolojik (Ort)':>16}  {'Patolojik (Min-Max)':>20}")
print("  " + "-" * 90)
for m in metrics:
    n_avg, n_min, n_max = avg(norm_rad, m)
    p_avg, p_min, p_max = avg(path_rad, m)
    print(f"  {m:<18}  {n_avg:>14.3f}  {n_min:>8.2f} - {n_max:<8.2f}  {p_avg:>16.3f}  {p_min:>8.2f} - {p_max:<8.2f}")

print("\n  PHASE OZET:")
print(f"  {'Metrik':<24}  {'Normal (Ort)':>14}  {'Patolojik (Ort)':>16}  {'Fark':>8}")
print("  " + "-" * 70)
for m in phase_metrics:
    n_avg = avg(norm_phase, m)[0]
    p_avg = avg(path_phase, m)[0]
    diff_pct = ((p_avg - n_avg) / n_avg) * 100
    print(f"  {m:<24}  {n_avg:>14.4f}  {p_avg:>16.4f}  {diff_pct:>+7.1f}%")

# --- Meningiom bolgesi vs diger bolgeler ---
print("\n" + "=" * 70)
print("  MENINGIOM BOLGESI vs DIGER BOLGELER (Sag T1+C)")
print("=" * 70)

men_slices = [r for r in rad if 9 <= int(r['slice']) <= 13]
other_slices = [r for r in rad if not (9 <= int(r['slice']) <= 13)]

men_phase = [p for p in phase if 9 <= int(p['slice']) <= 13]
other_phase = [p for p in phase if not (9 <= int(p['slice']) <= 13)]

print(f"\n  {'Metrik':<24}  {'Meningiom (9-13)':>18}  {'Diger Kesitler':>18}  {'Fark':>8}")
print("  " + "-" * 75)

for m in metrics:
    men_avg = sum(float(r[m]) for r in men_slices) / len(men_slices)
    oth_avg = sum(float(r[m]) for r in other_slices) / len(other_slices)
    diff_pct = ((men_avg - oth_avg) / oth_avg) * 100
    print(f"  {m:<24}  {men_avg:>18.3f}  {oth_avg:>18.3f}  {diff_pct:>+7.1f}%")

for m in phase_metrics:
    men_avg = sum(float(p[m]) for p in men_phase) / len(men_phase)
    oth_avg = sum(float(p[m]) for p in other_phase) / len(other_phase)
    diff_pct = ((men_avg - oth_avg) / oth_avg) * 100
    print(f"  {m:<24}  {men_avg:>18.4f}  {oth_avg:>18.4f}  {diff_pct:>+7.1f}%")

print()
