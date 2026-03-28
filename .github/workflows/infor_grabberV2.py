#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


material = "lead"#change per materia;
base_path = "C:/Users/ryan/OneDrive/Desktop/iron"

# Set file names and material properties for carbon
if material == "lead":
    det1_file = os.path.join(base_path, "LEAD_Det1.txt")
    det2_file = os.path.join(base_path, "LEAD_Det2.txt")
    X0 = 19.32  #change per material
    target_length = 1.5  
    expected_dEdx = 1.8 

output_dir = os.path.join(base_path, "output")
os.makedirs(output_dir, exist_ok=True)

# ---------------- BEAM PARAMETERS ----------------
p_mu = 4000.0  # MeV/c

# ---------------- USER-INPUT HIGHLAND PREDICTION ----------------
highland_projected = 5.77  # mrad (projected)

print("="*60)
print(f"{material.upper()} MULTIPLE SCATTERING ANALYSIS")
print("="*60)
print(f"Target thickness: {target_length} cm")
print(f"Radiation length X₀: {X0} cm")
print(f"USER HIGHLAND θ₀: {highland_projected:.4f} mrad (projected)")
print("="*60)

# ---------------- LOAD DATA ----------------
columns = ["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID","ParentID","Weight"]

print("\nLoading data...")
try:
    df1 = pd.read_csv(det1_file, sep=r'\s+', comment='#', names=columns)
    df2 = pd.read_csv(det2_file, sep=r'\s+', comment='#', names=columns)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Make sure {det1_file} and {det2_file} exist")
    exit(1)

print(f"Det1 entries: {len(df1)}")
print(f"Det2 entries: {len(df2)}")

# ---------------- FILTER AND ALIGN ----------------
df1 = df1[df1['TrackID'] == 1].copy()
df2 = df2[df2['TrackID'] == 1].copy()

df1 = df1.drop_duplicates(subset="EventID", keep='last')
df2 = df2.drop_duplicates(subset="EventID", keep='last')

df1 = df1.set_index("EventID").sort_index()
df2 = df2.set_index("EventID").sort_index()

common_events = df1.index.intersection(df2.index)
df1 = df1.loc[common_events]
df2 = df2.loc[common_events]

num_events = len(common_events)
print(f"Matched events: {num_events}")

# ---------------- CALCULATE SCATTERING ANGLES ----------------
def calculate_angles(px, py, pz):
    p_total = np.sqrt(px**2 + py**2 + pz**2)
    theta_x = np.arcsin(px / p_total)
    theta_y = np.arcsin(py / p_total)
    return theta_x, theta_y

theta_x1, theta_y1 = calculate_angles(df1['Px'].values, df1['Py'].values, df1['Pz'].values)
theta_x2, theta_y2 = calculate_angles(df2['Px'].values, df2['Py'].values, df2['Pz'].values)

# Calculate scattering angles
delta_x = (theta_x2 - theta_x1) * 1000
delta_y = (theta_y2 - theta_y1) * 1000
#if math denotes scale factor(did not in this project, the coder built this off of previous programs which did)
delta_x_scaled = delta_x
delta_y_scaled = delta_y 
delta_total_scaled = np.sqrt(delta_x_scaled**2 + delta_y_scaled**2)

valid = np.isfinite(delta_x_scaled) & np.isfinite(delta_y_scaled)
delta_x_scaled = delta_x_scaled[valid]
delta_y_scaled = delta_y_scaled[valid]
delta_total_scaled = delta_total_scaled[valid]

# ---------------- UNFILTERED DATA (ALL EVENTS) ----------------
delta_x_unfiltered = delta_x_scaled
delta_y_unfiltered = delta_y_scaled
delta_total_unfiltered = delta_total_scaled
num_unfiltered = len(delta_x_unfiltered)

# ---------------- 3-SIGMA FILTERED DATA ----------------
sigma_threshold = 3 * highland_projected
filter_mask = np.abs(delta_x_unfiltered) < sigma_threshold
delta_x_filtered = delta_x_unfiltered[filter_mask]
delta_y_filtered = delta_y_unfiltered[filter_mask]
delta_total_filtered = delta_total_unfiltered[filter_mask]
num_filtered = len(delta_x_filtered)
filter_percentage = (num_filtered / num_unfiltered) * 100

print(f"\nFiltering results:")
print(f"  Unfiltered events: {num_unfiltered}")
print(f"  Filtered events (3σ): {num_filtered}")
print(f"  Retained: {filter_percentage:.2f}%")

# ---------------- SIMULATED RMS (BOTH SETS) ----------------
# Unfiltered
sim_rms_x_unf = np.std(delta_x_unfiltered)
sim_rms_y_unf = np.std(delta_y_unfiltered)
sim_rms_total_unf = np.sqrt(sim_rms_x_unf**2 + sim_rms_y_unf**2)
percent_agreement_unf = (sim_rms_total_unf / highland_projected) * 100

# Filtered
sim_rms_x_filt = np.std(delta_x_filtered)
sim_rms_y_filt = np.std(delta_y_filtered)
sim_rms_total_filt = np.sqrt(sim_rms_x_filt**2 + sim_rms_y_filt**2)
percent_agreement_filt = (sim_rms_total_filt / highland_projected) * 100

# ---------------- ENERGY LOSS (using unfiltered events) ----------------
energy_loss = df1['Pz'].values - df2['Pz'].values
energy_loss_all = energy_loss[valid]
mean_energy_loss = np.mean(energy_loss_all)
dedx = mean_energy_loss / target_length

# ---------------- PLOT 1: HISTOGRAM (UNFILTERED) ----------------
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(delta_total_unfiltered, bins=80, alpha=0.7, color='orange', 
                                  edgecolor='black', linewidth=0.5, label='Simulated')
plt.axvline(highland_projected, color='red', linestyle='--', linewidth=2, 
            label=f'Highland: {highland_projected:.3f} mrad')
plt.axvline(sim_rms_total_unf, color='purple', linestyle=':', linewidth=2,
            label=f'Sim RMS: {sim_rms_total_unf:.3f} mrad')
plt.xlabel('Total Scattering Angle Δθ (mrad)')
plt.ylabel('Counts')
plt.title(f'{material.capitalize()} - Unfiltered Distribution (all events)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{material}_histogram_unfiltered.png"), dpi=300)
plt.show()

# ---------------- PLOT 2: DOT PLOT (UNFILTERED) ----------------
plt.figure(figsize=(8, 8))
plt.scatter(delta_x_unfiltered[::10], delta_y_unfiltered[::10], s=0.5, alpha=0.3, color='orange')
circle_highland = plt.Circle((0, 0), highland_projected, fill=False, color='red', 
                            linewidth=2, linestyle='--', label='Highland')
circle_sim = plt.Circle((0, 0), sim_rms_x_unf, fill=False, color='black', 
                       linewidth=2, linestyle=':', label='Sim RMS')
plt.gca().add_patch(circle_highland)
plt.gca().add_patch(circle_sim)
plt.xlabel('Δθ_x (mrad)')
plt.ylabel('Δθ_y (mrad)')
plt.title(f'{material.capitalize()} - Unfiltered 2D Pattern')
plt.axis('equal')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{material}_dotplot_unfiltered.png"), dpi=300)
plt.show()

# ---------------- PLOT 3: HISTOGRAM (FILTERED) ----------------
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(delta_total_filtered, bins=80, alpha=0.7, color='orange', 
                                  edgecolor='black', linewidth=0.5, label='Simulated (3σ filtered)')
plt.axvline(highland_projected, color='red', linestyle='--', linewidth=2, 
            label=f'Highland: {highland_projected:.3f} mrad')
plt.axvline(sim_rms_total_filt, color='purple', linestyle=':', linewidth=2,
            label=f'Sim RMS: {sim_rms_total_filt:.3f} mrad')
plt.axvline(sigma_threshold, color='orange', linestyle='-', linewidth=1.5,
            label=f'3σ cut: {sigma_threshold:.3f} mrad')
plt.xlabel('Total Scattering Angle Δθ (mrad)')
plt.ylabel('Counts')
plt.title(f'{material.capitalize()} - Filtered Distribution (3σ core)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{material}_histogram_filtered.png"), dpi=300)
plt.show()

# ---------------- PLOT 4: DOT PLOT (FILTERED) ----------------
plt.figure(figsize=(8, 8))
plt.scatter(delta_x_filtered[::10], delta_y_filtered[::10], s=0.5, alpha=0.3, color='orange')
circle_highland = plt.Circle((0, 0), highland_projected, fill=False, color='red', 
                            linewidth=2, linestyle='--', label='Highland')
circle_sim = plt.Circle((0, 0), sim_rms_x_filt, fill=False, color='black', 
                       linewidth=2, linestyle=':', label='Sim RMS')
circle_cut = plt.Circle((0, 0), sigma_threshold, fill=False, color='orange', 
                       linewidth=1.5, linestyle='-', label='3σ cut')
plt.gca().add_patch(circle_highland)
plt.gca().add_patch(circle_sim)
plt.gca().add_patch(circle_cut)
plt.xlabel('Δθ_x (mrad)')
plt.ylabel('Δθ_y (mrad)')
plt.title(f'{material.capitalize()} - Filtered 2D Pattern (3σ core)')
plt.axis('equal')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{material}_dotplot_filtered.png"), dpi=300)
plt.show()

# ---------------- SUMMARY FILE ----------------
summary_file = os.path.join(output_dir, f"{material}_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("="*70 + "\n")
    f.write(f"{material.upper()} MULTIPLE SCATTERING SUMMARY\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("CONFIGURATION\n")
    f.write(f"  Material: {material}\n")
    f.write(f"  Target thickness: {target_length} cm\n")
    f.write(f"  Radiation length X₀: {X0} cm\n")
    f.write(f"  Muon momentum: {p_mu} MeV/c\n")
    f.write(f"  Simulation scale factor: {scale_factor} (all data multiplied by 2)\n")
    f.write(f"  3σ filter threshold: {sigma_threshold:.4f} mrad\n\n")
    
    f.write("HIGHLAND PREDICTION (USER INPUT)\n")
    f.write(f"  θ₀ (projected): {highland_projected:.4f} mrad\n\n")
    
    f.write("STATISTICS\n")
    f.write(f"  Total events matched: {num_events}\n")
    f.write(f"  Valid events after angle calculation: {num_unfiltered}\n")
    f.write(f"  Events after 3σ filter: {num_filtered}\n")
    f.write(f"  Filter retention: {filter_percentage:.2f}%\n\n")
    
    f.write("-"*70 + "\n")
    f.write("UNFILTERED RESULTS (all events)\n")
    f.write("-"*70 + "\n")
    f.write(f"  σ_x: {sim_rms_x_unf:.4f} mrad\n")
    f.write(f"  σ_y: {sim_rms_y_unf:.4f} mrad\n")
    f.write(f"  Total RMS (√(σ_x²+σ_y²)): {sim_rms_total_unf:.4f} mrad\n")
    f.write(f"  Ratio Total / θ₀: {sim_rms_total_unf/highland_projected:.4f}\n")
    f.write(f"  Percent Agreement: {percent_agreement_unf:.2f}%\n\n")
    
    f.write("-"*70 + "\n")
    f.write("FILTERED RESULTS (3σ core)\n")
    f.write("-"*70 + "\n")
    f.write(f"  σ_x: {sim_rms_x_filt:.4f} mrad\n")
    f.write(f"  σ_y: {sim_rms_y_filt:.4f} mrad\n")
    f.write(f"  Total RMS (√(σ_x²+σ_y²)): {sim_rms_total_filt:.4f} mrad\n")
    f.write(f"  Ratio Total / θ₀: {sim_rms_total_filt/highland_projected:.4f}\n")
    f.write(f"  Percent Agreement: {percent_agreement_filt:.2f}%\n\n")
    
    f.write("-"*70 + "\n")
    f.write("ENERGY LOSS (unfiltered)\n")
    f.write("-"*70 + "\n")
    f.write(f"  Mean energy loss: {mean_energy_loss:.2f} MeV\n")
    f.write(f"  dE/dx: {dedx:.2f} MeV/cm\n")
    f.write(f"  Expected dE/dx for {material}: ~{expected_dEdx} MeV/cm\n")
    f.write("="*70 + "\n")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"Your Highland θ₀: {highland_projected:.4f} mrad")
print(f"\nUNFILTERED:")
print(f"  Total RMS: {sim_rms_total_unf:.4f} mrad")
print(f"  Agreement: {percent_agreement_unf:.2f}%")
print(f"\nFILTERED (3σ):")
print(f"  Total RMS: {sim_rms_total_filt:.4f} mrad")
print(f"  Agreement: {percent_agreement_filt:.2f}%")
print(f"  Retained: {filter_percentage:.2f}% of events")
print(f"\ndE/dx: {dedx:.2f} MeV/cm")
print(f"Summary saved to: {summary_file}")
print("="*70)
