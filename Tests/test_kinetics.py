import numpy as np

# ==============================================================================
# 1. TEST MATEMATICI INDIPENDENTI (Formule pura/Algoritmi del software)
# ==============================================================================

def test_df_f0_calculation():
    """
    Verifica il calcolo del delta F/F0:
    Formula: (F - F0) / F0 dove F0 è la baseline post sottrazione background.
    """
    raw_signal = np.array([100.0, 100.0, 150.0, 200.0, 100.0])
    bg_signal = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    
    # Sottrazione del background
    sub_bg = raw_signal - bg_signal
    # Baseline F0 calcolata sui primi 2 frame (valore = 90.0)
    f0 = np.mean(sub_bg[:2])
    
    df_f0 = (sub_bg - f0) / f0
    
    # Il valore baseline deve essere 0.0
    assert np.isclose(df_f0[0], 0.0)
    # Al frame 3 (valore 190.0): (190 - 90)/90 = 1.111...
    assert np.isclose(df_f0[3], (190.0 - 90.0) / 90.0)


def test_mad_robust_thresholding():
    """
    Verifica la stima del rumore tramite Median Absolute Deviation (MAD):
    Formula: sigma = MAD * 1.4826
    """
    noise = np.array([0.01, -0.01, 0.02, -0.02, 0.015, -0.015])
    median_val = np.median(noise)
    mad = np.median(np.abs(noise - median_val))
    sigma_robust = mad * 1.4826
    
    assert sigma_robust > 0
    assert np.isclose(median_val, 0.0)


def test_auc_trapezoidal_integration():
    """
    Verifica il calcolo dell'Area Sotto la Curva (AUC) per un singolo transiente.
    """
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # Picco triangolare con base 4 s e altezza 2
    signal = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    
    # Area teorica di un triangolo = (Base * Altezza) / 2 = (4 * 2) / 2 = 4.0
    auc = np.trapezoid(signal, time)
    
    assert np.isclose(auc, 4.0)


def test_decay_constant_tau():
    """
    Verifica il fit esponenziale del decadimento: F(t) = F0 * exp(-t / tau)
    Quando t = tau, l'ampiezza scende al 36.8% circa del valore iniziale (1/e).
    """
    tau_true = 1.5  # secondi
    t = np.linspace(0, 5, 100)
    amplitude = 1.0
    decay_signal = amplitude * np.exp(-t / tau_true)
    
    # Stima di tau prendendo il punto in cui il segnale cade a 1/e (~0.3678)
    target_val = amplitude / np.e
    idx = np.argmin(np.abs(decay_signal - target_val))
    tau_estimated = t[idx]
    
    # Tolleranza stretta sul valore di tau stimato
    assert np.isclose(tau_estimated, tau_true, atol=0.1)


def test_rate_of_rise():
    """
    Verifica la velocità massima di salita del picco (Rate of Rise / Slope max).
    """
    time = np.array([0.0, 0.1, 0.2, 0.3])
    signal = np.array([0.0, 0.2, 1.0, 1.1])
    
    # Derivata prima (delta_signal / delta_time)
    slopes = np.diff(signal) / np.diff(time)
    max_rate_of_rise = np.max(slopes)
    
    # La pendenza massima è tra t=0.1 e t=0.2 -> (1.0 - 0.2)/0.1 = 8.0
    assert np.isclose(max_rate_of_rise, 8.0)


def test_correlation_matrix_properties():
    """
    Verifica le proprietà fondamentali della matrice di sincronia di Pearson:
    1. Diagonale principale uguale a 1.0
    2. Matrice perfettamente simmetrica
    """
    # 3 neuroni con segnali sintetici
    neuron1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    neuron2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # perfettamente correlato con 1
    neuron3 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # anti-correlato con 1

    dataset = np.vstack([neuron1, neuron2, neuron3])
    corr_matrix = np.corrcoef(dataset)

    # 1. Autocorrelazione sulla diagonale = 1
    assert np.allclose(np.diag(corr_matrix), 1.0)
    # 2. Correlazione tra neuron1 e neuron2 = 1
    assert np.isclose(corr_matrix[0, 1], 1.0)
    # 3. Correlazione tra neuron1 e neuron3 = -1
    assert np.isclose(corr_matrix[0, 2], -1.0)
    # 4. Simmetria della matrice (A == A^T)
    assert np.allclose(corr_matrix, corr_matrix.T)


def test_event_coincidence_rate_ecr():
    """
    Verifica la coincidenza temporale degli eventi (ECR) tra due vettori binari di picchi.
    """
    # Picchi rilevati ai vari frame per 2 neuroni
    peaks_neuron_A = np.array([0, 1, 0, 0, 1, 0])
    peaks_neuron_B = np.array([0, 1, 0, 0, 0, 1])
    
    # Coincidenze esatte (entrambi 1 nello stesso frame)
    coincidences = np.sum((peaks_neuron_A == 1) & (peaks_neuron_B == 1))
    total_events_A = np.sum(peaks_neuron_A)
    
    ecr = coincidences / total_events_A if total_events_A > 0 else 0.0
    
    # Coincidenza su 1 dei 2 picchi di A -> 1/2 = 0.5
    assert np.isclose(ecr, 0.5)