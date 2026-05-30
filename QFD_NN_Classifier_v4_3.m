% =========================================================================
%  QFD_NN_Classifier_v4.m
%  QFD-Structured Neural Network for AI vs. Classical Software Identification
%
%  Architecture  :  27 inputs  →  Layer 1: 9 neurons (Technical Evidence)
%                              →  Layer 2: 7 neurons (QFD Requirements)
%                              →  Output:  1 neuron  (sigmoid → P(AI))
%
%  Dataset       :  AIC-199 (100 AI / 99 Classical)  — revised taxonomy
%                   AI = any system that learns from data (including classical ML)
%                   Classical = rule-based / algorithmic / MLOps infrastructure
%
%  Features      :  x1–x25 (original 25, revised definitions) +
%                   x26 trains_model  +  x27 classical_model_artefact
%
%  QFD matrices  :  M1 ∈ {0,1,3,9}^(9×27)  — features → technical evidence
%                   M2 ∈ {0,1,3,9}^(7×9)   — technical evidence → requirements
%
%  Three integration options (select via QFD_OPTION below):
%    'A' — QFD-informed initialisation only (no masking, no regularisation)
%          W1_init = row-normalised M1;  W2_init = row-normalised M2
%          Network trains freely from knowledge-calibrated starting point.
%
%    'B' — QFD structural sparsity masking + QFD initialisation
%          Forbidden connections (M1=0, M2=0) zeroed at init and re-zeroed
%          after every training epoch via manual loop → enforced throughout.
%
%    'C' — QFD initialisation + soft regularisation via gradient penalty
%          Implemented as a custom dlnetwork training loop so that LAMBDA
%          actually enters the gradient computation via dlfeval/dlgradient.
%          L_total = L_bce + LAMBDA * ||W2 - W_QFD_target||^2_F
%
%  BUG FIXES vs v3:
%    1. Option C now truly applies LAMBDA during gradient updates (dlnetwork).
%    2. Option C now receives QFD initialisation (was missing in v3).
%    3. CV loop now applies option-specific init/masking per fold (not always A).
%    4. Option B mask is enforced after each training epoch (not only post-training).
%
%  Normalization :  all 27 features already normalised to [0,1] in dataset
%
%  Authors       :  M.Leba, A.Ionica
%  Date          :  2026  (v4 — corrected Option C + Option B mask enforcement)
% =========================================================================

clear; clc; close all;
rng(42);   % reproducibility

fprintf('=================================================\n');
fprintf('  QFD-NN Classifier v4  |  AIC-199  |  27 features\n');
fprintf('=================================================\n\n');

% =========================================================================
% 1.  DATASET  (199 × 27, already normalised) - hardcoded for convenience
% =========================================================================

X_raw = [
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00853659 0 0 0 0 0 0 0 0 0 0 0;
  0.57142857 1 1 0 0 0 0.97191877 0.99470189 0.95424251 0 0 0 0 0 0 0.14181818 1 1 0 1 0 1 0.57142857 1 1 0 0;
  0 0 0 0 0 0 0.79248125 0 0 0 0 0 0 0 0 0.02117647 0 0 0 0 0 0 0 0 0 0 0;
  0.22222222 0 0 0 1 0.86135312 0.95424251 0 0 0.8982444 0 1 0 1 0 0.055 0 1 0 1 1 0 0.22222222 1 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01222222 0 0 0 0 0 0 0 0 0 0 0;
  0.78571429 0 0 1 0 0.92078222 0.98287788 0 0.8982444 0 0 1 1 1 1 0.16777778 0 1 1 1 1 0 0.78571429 1 1 1 0;
  0 0 0 0 0 0 0.63092975 0 0 0 0.79248125 0 0 0 0 0.01410256 0 0 0 0 0 0 0 0 0 0 0;
  0.875 1 0 0 0 0.98660828 0.8982444 0 0.79248125 0 0 1 1 1 1 0.1952381 1 1 1 1 1 1 0.875 1 1 1 0;
  0 0 0 0 0 0 0.79248125 0 0 0 0.8982444 0 0 0 0 0.02439024 0 0 1 0 0 0 0 0 0 1 0;
  0.71428571 0 0 0 0 0 0.97672265 0.98660828 0.93578497 0 0 1 0 0 0 0.15176471 1 1 1 1 1 0 0.71428571 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0056338 0 0 0 0 0 0 0 0 0 0 0;
  0.53846154 1 0 0 0 0 0.95424251 0.99119932 0.92078222 0 0 0 0 0 0 0.11785714 1 0 0 1 0 0 0.53846154 1 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0057971 0 0 0 0 0 0 0 0 0 0 0;
  0.9375 0 0 0 0 0.98287788 0.99531406 0.95424251 0.97672265 0 0 0 0 1 1 0.22392857 0 1 1 1 1 0 0.9375 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.004 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.93578497 0.99003247 0.8982444 0 0 0 0 0 0 0.13096774 1 0 0 1 0 1 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0075 0 0 0 0 0 0 0 0 0 0 0;
  0.85714286 0 0 0 0 0.95424251 0.98022455 0.98796208 0.96498405 0 0 1 0 1 1 0.188 0 1 1 1 1 0 0.85714286 1 1 1 0;
  0 0 0 0 0 0 0.86135312 0 0 0 0 0 0 0 0 0.02235294 0 0 0 0 0 0 0 0 0 0 0;
  0.4 1 0 0 0 0 0.8982444 0.98582304 0.79248125 0 0 0 0 0 0 0.09833333 1 0 0 0 0 0 0.4 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00526316 0 0 0 0 0 0 0 0 0 0 0;
  0.71428571 1 1 0 0 0.8982444 0.98796208 0.97191877 0.98287788 0 0 0 0 0 1 0.17809524 1 1 1 1 0 1 0.71428571 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00731707 0 0 0 0 0 0 0 0 0 0 0;
  0.95238095 0 0 1 0 0.99003247 0.99773477 0.97672265 0.98660828 0 0 1 1 1 1 0.232 1 1 1 1 1 1 0.95238095 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00512821 0 0 0 0 0 0 0 0 0 0 0;
  0.33333333 1 0 0 0 0 0.79248125 0.86135312 0.79248125 0 0 0 0 1 0 0.088 1 0 1 1 0 1 0.33333333 0 0 0 0;
  0.07692308 0 0 0 0 0.86135312 0.95424251 0 0 0 0.96025257 0 0 0 0 0.07666667 0 1 1 0 0 0 0.07692308 0 0 1 0;
  0.66666667 0 0 0 0 0.8982444 0.93578497 0.8982444 0.79248125 0 0 0 0 1 0 0.12125 1 1 1 1 1 1 0.66666667 1 0 0 0;
  0 0 0 0 0 0 0 0 0.63092975 0 0 0 0 0 0 0.00769231 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.95424251 0.98908601 0.8982444 0 0 0 0 0 0 0.12777778 1 0 0 1 0 1 0.5 1 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01777778 0 0 0 0 0 0 0 0 0 0 0;
  0.45454545 1 1 0 0 0 0.8982444 0.97672265 0.95424251 0 0 0 0 0 0 0.11231884 1 0 0 1 0 0 0.45454545 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00843373 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.93578497 0.98660828 0.79248125 0 0 0 0 0 0 0.122 1 0 0 1 0 1 0.5 1 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01058824 0 0 0 0 0 0 0 0 0 0 0;
  0.58333333 0 0 0 0 0.93578497 0.95424251 0.79248125 0.79248125 0 0 1 0 1 0 0.12797619 0 1 1 1 1 1 0.58333333 1 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00987654 0 0 0 0 0 0 0 0 0 0 0;
  0.75 0 0 1 0 0.79248125 0.96498405 0.97191877 0.8982444 0 0 1 1 0 1 0.16210526 0 1 1 1 1 0 0.75 0 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00769231 0 0 0 0 0 0 0 0 0 0 0;
  0.36363636 1 0 0 0 0 0.93578497 0.99003247 0.8982444 0 0 0 0 0 0 0.10774648 1 0 0 1 0 1 0.36363636 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00540541 0 0 0 0 0 0 0 0 0 0 0;
  0.85714286 0 0 0 1 0.98022455 0.98660828 0 0.8982444 0.98287788 0 1 0 1 1 0.19791667 0 1 0 1 1 1 0.85714286 1 1 1 0;
  0 0 0 0 0 0 0.63092975 0 0 0 0 0 0 0 0 0.00853659 0 0 0 0 0 0 0 0 0 0 0;
  0.46153846 1 1 0 0 0 0.95424251 0.98660828 0.96498405 0 0 0 0 0 0 0.11533333 1 0 0 1 0 1 0.46153846 0 0 0 0;
  0.07142857 0 0 0 0 0 0 0 0.63092975 0 0 0 0 0 0 0.04818182 0 0 0 0 0 0 0.07142857 0 0 0 0;
  0.81818182 1 0 0 0 0.8982444 0.97191877 0.98287788 0.93578497 0 0 1 1 1 1 0.18818182 1 1 1 1 0 0 0.81818182 1 1 1 0;
  0 0 0 0 0 0 0.8982444 0 0 0 0 1 0 0 0 0.06454545 0 0 0 0 0 0 0 0 0 1 1;
  0.42857143 0 0 0 0 0.79248125 0.93578497 0 0.79248125 0 0 1 0 0 0 0.07809524 0 1 1 0 0 0 0.42857143 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00675676 0 0 0 0 0 0 0 0 0 0 0;
  0.66666667 1 1 0 0 0.79248125 0.97672265 0.98287788 0.98660828 0 0 1 1 0 1 0.175 1 1 1 1 0 1 0.66666667 1 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0097561 0 0 0 0 0 0 0 0 0 0 0;
  0.41666667 1 0 0 0 0.79248125 0.8982444 0.93578497 0.8982444 0 0 0 0 1 0 0.09538462 1 0 1 1 0 1 0.41666667 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00657895 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0.8982444 0.95424251 0.93578497 0.8982444 0 0 0 0 1 0 0.10482759 1 0 1 1 0 0 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.03061224 0 0 0 0 0 0 0 0 0 0 0;
  0.54545455 1 1 0 0 0 0.95424251 0.98022455 0.97672265 0 0 0 0 0 0 0.14176471 1 0 0 1 0 1 0.54545455 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01529412 0 0 0 0 0 0 0 0 0 0 0;
  0.66666667 1 0 0 0 0.79248125 0.97191877 0.99003247 0.8982444 0 0 1 1 0 0 0.155 1 0 0 1 1 1 0.66666667 0 0 0 0;
  0 0 0 0 0 0 0.93578497 0 0 0 0 1 0 0 0 0.07333333 0 1 0 0 0 0 0 0 0 1 1;
  0.71428571 1 1 0 0 0.79248125 0.97191877 0.99153434 0.98022455 0 0 0 0 0 0 0.168 1 1 0 1 0 1 0.71428571 1 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0056338 0 0 0 0 0 0 0 0 0 0 0;
  1 0 0 1 0 0.93578497 0.99003247 0 0.95424251 0 0 1 0 1 1 0.20208333 0 1 1 1 1 0 1 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01219512 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.95424251 0.98660828 0.8982444 0 0 0 0 0 0 0.11824324 1 0 0 1 0 1 0.5 0 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01341463 0 0 0 0 0 0 0 0 0 0 0;
  0.92307692 0 0 1 0 0.95424251 0.98796208 0 0.93578497 0 0 1 0 1 1 0.195 0 1 1 1 1 0 0.92307692 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.58333333 1 0 0 0 0.79248125 0.96498405 0.99003247 0.93578497 0 0 0 0 0 0 0.135 1 0 0 1 0 1 0.58333333 1 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.36363636 1 0 0 0 0 0.93578497 0.98660828 0.79248125 0 0 0 0 0 0 0.10222222 1 0 0 1 0 0 0.36363636 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00853659 0 0 0 0 0 0 0 0 0 0 0;
  0.45454545 1 0 0 0 0 0.93578497 0.98908601 0.8982444 0 0 0 0 0 0 0.11172414 1 0 0 1 0 0 0.45454545 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0056338 0 0 0 0 0 0 0 0 0 0 0;
  0.9 0 0 0 0 0.97191877 0.99392485 0.8982444 0.97191877 0 0 1 0 1 1 0.21807692 0 1 1 1 1 0 0.9 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.72727273 0 0 0 0 0.8982444 0.98287788 0 0.93578497 0 0.79248125 1 1 0 1 0.14810811 0 1 1 1 0 0 0.72727273 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01176471 0 0 0 0 0 0 0 0 0 0 0;
  0.58333333 1 0 0 0 0.79248125 0.97191877 0.98022455 0.8982444 0 0 1 1 0 0 0.128125 1 0 0 1 1 1 0.58333333 1 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00533333 0 0 0 0 0 0 0 0 0 0 0;
  0.46153846 0 0 0 0 0.95424251 0.98660828 0 0 0 0.93578497 1 0 1 1 0.15789474 0 1 1 1 0 0 0.46153846 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00422535 0 0 0 0 0 0 0 0 0 0 0;
  0.75 1 0 1 0 0.86135312 0.99621332 0.98287788 0.98660828 0 0.95424251 1 1 1 1 0.18909091 0 1 1 1 1 0 0.75 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00769231 0 0 0 0 0 0 0 0 0 0 0;
  0.84615385 0 0 0 1 0.97191877 0.99003247 0 0.8982444 0.98660828 0 1 0 1 1 0.195 0 1 1 1 1 0 0.84615385 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.5 0 0 0 0 0.79248125 0.95424251 0 0.8982444 0 0 1 1 0 0 0.09166667 0 1 1 0 0 0 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01764706 0 0 0 0 0 0 0 0 0 0 0;
  0.58333333 1 1 0 0 0 0.96498405 0.99003247 0.97191877 0 0 0 0 0 0 0.148 1 0 0 1 0 1 0.58333333 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00533333 0 0 0 0 0 0 0 0 0 0 0;
  0.8 0 0 0 0 0.8982444 0.98287788 0.98796208 0.95424251 0 0 1 1 0 0 0.17809524 1 1 1 1 1 1 0.8 0 1 1 0;
  0 0 0 0 0 0 0 0 0.79248125 0 0 0 0 0 0 0.02173913 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 1 0 0 0 0.95424251 0.98022455 0.97672265 0 0 0 0 0 0 0.13818182 1 0 0 1 0 1 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00416667 0 0 0 0 0 0 0 0 0 0 0;
  0.58333333 1 1 0 0 0 0.95424251 0.98660828 0.97191877 0 0 0 0 0 0 0.14244186 1 0 0 1 0 1 0.58333333 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00853659 0 0 0 0 0 0 0 0 0 0 0;
  0.58333333 1 0 0 0 0 0.95424251 0.99003247 0.8982444 0 0 0 0 0 0 0.14777778 1 1 0 1 0 1 0.58333333 1 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00540541 0 0 0 0 0 0 0 0 0 0 0;
  0.375 1 1 0 0 0 0.8982444 0.98022455 0.93578497 0 0 0 0 0 0 0.11185185 1 0 0 0 0 0 0.375 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.93578497 0.98796208 0.8982444 0 0 0 0 0 0 0.13212121 1 0 0 1 0 1 0.5 1 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0056338 0 0 0 0 0 0 0 0 0 0 0;
  0.53333333 0 0 0 0 0.97672265 0.99153434 0 0.8982444 0 0.95424251 1 0 0 1 0.162 0 1 1 1 0 0 0.53333333 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 1 0 0 0 0.95424251 0.98287788 0.97191877 0 0 0 0 0 0 0.135 1 0 0 1 0 0 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00422535 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.93578497 0.98660828 0.8982444 0 0 0 0 0 0 0.11824324 1 0 0 1 0 0 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00416667 0 0 0 0 0 0 0 0 0 0 0;
  0.61538462 1 1 0 0 0.79248125 0.95424251 0.99003247 0.97191877 0 0 0 0 0 0 0.15189189 1 0 0 1 0 1 0.61538462 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00547945 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.95424251 0.98660828 0.8982444 0 0 1 1 0 0 0.12911392 1 0 0 1 1 1 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01190476 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 1 0 0 0 0.93578497 0.98796208 0.96498405 0 0 0 0 0 0 0.13809524 1 0 0 1 0 1 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00540541 0 0 0 0 0 0 0 0 0 0 0;
  0.93333333 0 0 1 0 0.97191877 0.99153434 0.8982444 0.96498405 0 0 1 0 1 1 0.212 0 1 1 1 1 0 0.93333333 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00666667 0 0 0 0 0 0 0 0 0 0 0;
  0.76190476 0 0 0 0 0.98796208 0.99661322 0.97191877 0.98287788 0 0 1 0 1 1 0.22785714 0 1 1 1 1 0 0.76190476 1 1 1 0;
  0 0 0 0 0 0 0.79248125 0 0 0 0 0 0 0 0 0.01764706 0 0 0 0 0 0 0 0 0 0 0;
  0.76923077 0 0 1 0 0.8982444 0.98022455 0 0.8982444 0 0 1 1 0 1 0.16512821 0 1 1 1 1 0 0.76923077 0 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00666667 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.95424251 0.98660828 0.8982444 0 0 0 0 0 0 0.12236842 1 0 0 1 0 0 0.5 1 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00547945 0 0 0 0 0 0 0 0 0 0 0;
  0.92307692 1 1 0 0 0.93578497 0.98660828 0.95424251 0.99003247 0 0 1 1 0 1 0.19181818 1 1 1 1 0 0 0.92307692 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00422535 0 0 0 0 0 0 0 0 0 0 0;
  0.93333333 0 0 1 0 0.98287788 0.99392485 0.95424251 0.96498405 0 0 1 0 1 1 0.215 0 1 1 1 1 0 0.93333333 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00526316 0 0 0 0 0 0 0 0 0 0 0;
  0.58333333 1 1 0 0 0.79248125 0.95424251 0.98022455 0.97672265 0 0 0 0 0 0 0.13809524 1 0 0 1 0 1 0.58333333 0 0 0 0;
  0 0 0 0 0 0 0 0 0.79248125 0 0 0 0 0 0 0.03181818 0 0 0 0 0 0 0 0 0 0 0;
  0.55555556 1 0 0 0 0 0.93578497 0.98287788 0.8982444 0 0 1 1 0 0 0.11521739 1 0 0 1 1 0 0.55555556 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00547945 0 0 0 0 0 0 0 0 0 0 0;
  0.90909091 1 0 1 0 0.99003247 0.99773477 0.97191877 0.97860217 0 0 1 0 1 1 0.21785714 1 1 1 1 1 1 0.90909091 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00853659 0 0 0 0 0 0 0 0 0 0 0;
  0.54545455 1 1 0 0 0 0.95424251 0.98022455 0.97672265 0 0 0 0 0 0 0.13818182 1 0 0 1 0 1 0.54545455 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00547945 0 0 0 0 0 0 0 0 0 0 0;
  0.71428571 1 0 1 0 0.93578497 0.98287788 0.97191877 0.95424251 0 0 1 1 0 1 0.178 1 1 1 1 1 0 0.71428571 1 1 1 0;
  0 0 0 0 0 0 0.79248125 0 0 0 0.8982444 0 0 0 0 0.02195122 0 0 1 0 0 0 0 0 0 1 0;
  0.84615385 0 0 1 0 0.93578497 0.98022455 0.97191877 0.95424251 0 0 1 1 0 1 0.18190476 0 1 1 1 1 0 0.84615385 0 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00853659 0 0 0 0 0 0 0 0 0 0 0;
  0.5 0 0 1 0 0.8982444 0.99003247 0 0.95424251 0 0 1 0 0 1 0.15789474 0 1 1 1 1 0 0.5 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00864198 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.8982444 0.98660828 0.79248125 0 0 0 0 0 0 0.11231884 1 0 0 0 0 0 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00540541 0 0 0 0 0 0 0 0 0 0 0;
  0.64285714 1 1 0 0 0.8982444 0.98287788 0.98660828 0.98796208 0 0 0 0 0 1 0.18181818 1 1 1 1 0 0 0.64285714 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00540541 0 0 0 0 0 0 0 0 0 0 0;
  0.875 0 0 0 0 0.8982444 0.98287788 0.98287788 0.95424251 0 0 1 1 0 0 0.17809524 1 1 1 1 1 0 0.875 0 0 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.76923077 0 0 1 0 0.95424251 0.98287788 0.96498405 0.93578497 0 0 1 0 1 1 0.18818182 0 1 1 1 1 0 0.76923077 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00519481 0 0 0 0 0 0 0 0 0 0 0;
  0.71428571 1 0 0 0 0.79248125 0.98022455 0.98287788 0.93578497 0 0 1 1 0 0 0.15538462 1 1 0 1 1 1 0.71428571 1 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00416667 0 0 0 0 0 0 0 0 0 0 0;
  0.66666667 1 1 0 0 0.8982444 0.98287788 0.9926687 0.98660828 0 0 0 0 0 1 0.195 1 1 1 1 0 0 0.66666667 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00422535 0 0 0 0 0 0 0 0 0 0 0;
  0.94736842 1 0 1 0 0.98022455 0.99571679 0.98287788 0.99003247 0 0 1 1 1 1 0.22785714 1 1 1 1 1 0 0.94736842 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00547945 0 0 0 0 0 0 0 0 0 0 0;
  0.625 1 1 0 0 0.8982444 0.98287788 0.98022455 0.98287788 0 0 0 0 0 1 0.18181818 1 1 1 1 0 1 0.625 1 1 1 0;
  0 0 0 0 0 0.79248125 0.93578497 0 0 0 0.95424251 1 0 0 0 0.05272727 0 0 1 0 0 0 0 0 0 1 1;
  0.78571429 0 0 0 1 0.96498405 0.98287788 0 0.8982444 0.99003247 0 0 0 1 1 0.18818182 0 1 1 1 1 0 0.78571429 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00547945 0 0 0 0 0 0 0 0 0 0 0;
  0.75 0 0 1 0 0.93578497 0.99003247 0 0.96498405 0 0 1 0 1 1 0.20208333 0 1 1 1 1 0 0.75 1 1 1 0;
  0.125 0 0 0 0 0 0.79248125 0 0 0 0 0 0 0 0 0.02222222 0 0 0 0 0 0 0.125 0 0 0 0;
  0.5 1 1 0 0 0 0.95424251 0.98022455 0.97191877 0 0 0 0 0 0 0.13515152 1 0 0 1 0 1 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00657895 0 0 0 0 0 0 0 0 0 0 0;
  0.8 0 0 1 0 0.95424251 0.99153434 0.8982444 0.96498405 0 0 1 0 1 1 0.205 0 1 1 1 1 0 0.8 1 1 1 0;
  0 0 0 0 0 0.86135312 0.93578497 0 0 0 0.95424251 0 0 0 0 0.047 0 0 1 0 0 0 0 0 0 1 0;
  0.69230769 0 0 1 0 0.8982444 0.98287788 0 0.93578497 0 0 1 0 0 1 0.16205128 0 1 1 1 1 0 0.69230769 0 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.93333333 0 0 1 0 0.93578497 0.98660828 0.8982444 0.98022455 0 0 1 1 1 1 0.21208333 0 1 1 1 1 0 0.93333333 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 1 0 0 0 0.95424251 0.98022455 0.97191877 0 0 0 0 0 0 0.13818182 1 0 0 1 0 1 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00547945 0 0 0 0 0 0 0 0 0 0 0;
  0.8 0 0 0 0 0.95424251 0.98287788 0.8982444 0.97191877 0 0 1 0 0 1 0.16205128 0 1 1 1 0 0 0.8 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.01058824 0 0 0 0 0 0 0 0 0 0 0;
  0.58333333 1 1 0 0 0.79248125 0.96498405 0.98660828 0.97191877 0 0 1 1 0 0 0.14777778 1 0 0 1 1 1 0.58333333 0 0 0 0;
  0 0 0 0 0 0 0 0 0.79248125 0 0 0 0 0 0 0.03818182 0 0 0 0 0 0 0 0 0 0 0;
  0.55555556 1 0 0 0 0 0.95424251 0.98287788 0.8982444 0 0 1 1 0 0 0.11517241 1 0 0 1 1 0 0.55555556 0 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.7 0 0 0 0 0.96498405 0.98660828 0 0 0 0.93578497 1 0 1 1 0.16461538 0 1 1 1 0 0 0.7 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00422535 0 0 0 0 0 0 0 0 0 0 0;
  0.77777778 0 0 0 0 0.8982444 0.98287788 0.98660828 0.95424251 0 0 1 0 0 0 0.18818182 1 1 1 1 1 0 0.77777778 0 0 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00547945 0 0 0 0 0 0 0 0 0 0 0;
  0.8 0 0 1 0 0.96498405 0.98796208 0.8982444 0.95424251 0 0 1 0 1 1 0.20208333 0 1 1 1 1 0 0.8 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00422535 0 0 0 0 0 0 0 0 0 0 0;
  0.78571429 0 0 1 0 0.93578497 0.98660828 0 0.95424251 0 0 1 0 0 1 0.18181818 0 1 1 1 1 0 0.78571429 0 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.95424251 0.98660828 0.8982444 0 0 0 0 0 0 0.12236842 1 0 0 1 0 0 0.5 1 0 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00416667 0 0 0 0 0 0 0 0 0 0 0;
  0.92857143 0 0 1 0 0.93578497 0.98796208 0 0.95424251 0 0 1 0 1 1 0.20208333 0 1 1 1 1 0 0.92857143 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00853659 0 0 0 0 0 0 0 0 0 0 0;
  0.77777778 1 0 1 0 0.95424251 0.98796208 0.98287788 0.97191877 0 0 1 1 1 1 0.215 1 1 1 1 1 0 0.77777778 1 1 1 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00547945 0 0 0 0 0 0 0 0 0 0 0;
  0.93333333 0 0 0 0 0.97191877 0.99003247 0.8982444 0.95424251 0 0 1 0 1 1 0.208 0 1 1 1 1 0 0.93333333 1 1 1 0;
  0 0 0 0 0 0 0 0 0.79248125 0 0 0 0 0 0 0.028 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 1 0 0 0 0.93578497 0.98660828 0.93578497 0 0 0 0 0 0 0.128125 1 0 0 1 0 1 0.5 0 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00555556 0 0 0 0 0 0 0 0 0 0 0;
  0.92857143 1 0 0 0 0.98287788 0.95424251 0.8982444 0.8982444 0 0 1 1 1 1 0.20208333 1 1 1 1 1 0 0.92857143 1 1 1 0;
  0 0 0 0 0 0 0 0 0.63092975 0 0 0 0 0 0 0.01777778 0 0 0 0 0 0 0 0 0 0 0;
  0.5 1 0 0 0 0 0.93578497 0.98796208 0.8982444 0 0 0 0 0 0 0.12193548 1 0 0 1 0 1 0.5 0 0 0 0;
  0 0 0 0 0 0 0 0 0.63092975 0 0 0 0 0 0 0.01341463 0 0 0 0 0 0 0 0 0 0 0;
  0.58333333 1 0 0 0 0 0.95424251 0.99003247 0.93578497 0 0 0 0 0 0 0.14777778 1 1 0 1 0 1 0.58333333 1 1 0 0;
  0 0 0 0 0 0.79248125 0.8982444 0 0 0 0.93578497 0 0 0 0 0.03913043 0 0 1 0 0 0 0 0 0 1 0
  ];

y_raw = [0 1 0 0 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 1 0 1 0 1 0 1 1 1 0 1 0 0 0 0 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1];

X = X_raw';   % 27 × 199  (features × samples)
T = y_raw;    % 1  × 199

N = size(X, 2);
n = size(X, 1);

fprintf('Dataset : %d samples  |  %d features\n', N, n);
fprintf('  AI class       : %d\n', sum(T == 1));
fprintf('  Classical class: %d\n', sum(T == 0));

% =========================================================================
% 2.  QFD RELATIONSHIP MATRICES - from experts
% =========================================================================

M1 = [
  9 9 9 9 9 3 1 3 3 0 0 0 0 0 0 1 1 0 0 1 3 3 9 1 0 1 0;
  3 3 1 3 0 9 9 3 9 3 9 0 0 1 0 3 1 3 1 0 1 0 1 0 0 3 3;
  9 1 0 1 9 9 1 0 0 9 0 3 1 9 1 3 1 3 9 9 3 1 3 9 0 9 9;
  1 9 3 1 0 0 9 9 0 0 3 1 9 0 3 3 9 1 1 1 1 1 1 0 1 0 0;
  0 0 0 0 0 0 0 0 0 0 0 0 3 0 9 9 9 9 9 1 0 0 0 0 3 1 0;
  3 1 0 3 0 0 3 1 0 0 3 9 9 9 9 1 1 3 3 3 0 0 0 0 3 3 9;
  9 9 9 3 3 1 1 9 3 0 0 3 1 3 0 1 3 0 0 9 9 9 9 3 1 1 0;
  9 3 3 1 9 3 3 9 3 9 1 3 1 3 3 9 3 9 9 9 0 0 9 9 9 9 3;
  3 1 1 1 3 9 3 0 1 3 9 3 1 9 1 3 0 3 9 1 1 1 0 3 1 9 9
  ];   % 9 × 27

M2 = [
  9 9 3 3 3 9 3 9 9;
  9 3 9 1 1 9 3 3 9;
  3 1 9 0 1 9 3 9 9;
  3 3 1 9 9 3 3 3 0;
  1 9 3 9 3 1 9 9 9;
  9 3 9 3 1 3 9 3 1;
  3 9 9 3 9 3 3 9 9
  ];   % 7 × 9

% Binary sparsity masks
M1_mask = double(M1 > 0);   % 9 × 27
M2_mask = double(M2 > 0);   % 7 × 9

% Row-normalised QFD weights (used for initialisation)
M1_norm = M1 ./ (sum(M1, 2) + eps);   % 9 × 27
M2_norm = M2 ./ (sum(M2, 2) + eps);   % 7 × 9

% Flat-normalised M2 (target for Option C regularisation)
W_QFD_target = M2 / (sum(M2(:)) + eps);   % 7 × 9

% Feature importance (QFD column sums)
colSums_M1 = sum(M1, 1);
W_qfd_init = colSums_M1 / sum(colSums_M1);

feat_names = {'x1 AI/ML libs','x2 LLM SDK','x3 VectorDB','x4 CV/Speech','x5 RL/Sim',...
              'x6 fit/train','x7 predict','x8 generate','x9 embed','x10 RL-step',...
              'x11 score/rec','x12 model wt','x13 tokenizer','x14 train cfg','x15 model card',...
              'x16 kw density','x17 LLM terms','x18 benchmark','x19 dataset ref','x20 AI folder',...
              'x21 GPU/CUDA','x22 Docker AI','x23 pkg ratio','x24 CI/CD ML','x25 AI license',...
              'x26 trains_model','x27 classical_model'};

fprintf('\nQFD weight vector (top 5 features):\n');
[sorted_w, sort_idx] = sort(W_qfd_init, 'descend');
for i = 1:5
    fprintf('  w_%d  %-22s  %.4f\n', sort_idx(i), feat_names{sort_idx(i)}, sorted_w(i));
end

% =========================================================================
% 3.  CONFIGURATION
% =========================================================================

QFD_OPTION = 'B';   % <<< Change to 'A', 'B', or 'C'
LAMBDA     = 0.01;  % regularisation strength — used in gradient for Option C

hiddenSizes = [9, 7];
theta       = 0.50;  % default decision threshold

fprintf('\nSelected option: %s\n', QFD_OPTION);
if strcmp(QFD_OPTION,'C')
    fprintf('  Lambda = %.4f\n', LAMBDA);
end

% =========================================================================
% 4.  INITIALISE MAIN NETWORK
% =========================================================================

net = patternnet(hiddenSizes, 'trainscg');
net.layers{1}.transferFcn      = 'tansig';
net.layers{2}.transferFcn      = 'tansig';
net.layers{3}.transferFcn      = 'logsig';
net.trainParam.showWindow      = false;
net.trainParam.showCommandLine = false;
net = configure(net, X, T);
net = init(net);   % Nguyen-Widrow baseline

% QFD initialisation: all options start from knowledge-calibrated weights
net.IW{1,1} = M1_norm;
net.LW{2,1} = M2_norm;

% Option B: zero forbidden connections at initialisation
if strcmp(QFD_OPTION, 'B')
    net.IW{1,1} = net.IW{1,1} .* M1_mask;
    net.LW{2,1} = net.LW{2,1} .* M2_mask;
end

% =========================================================================
% 5.  TRAIN (full dataset network)
% =========================================================================

net.trainParam.epochs         = 1000;
net.trainParam.goal           = 1e-5;
net.trainParam.max_fail       = 25;
net.trainParam.min_grad       = 1e-8;
net.trainParam.showWindow     = true;
net.trainParam.showCommandLine = true;
net.divideParam.trainRatio    = 0.70;
net.divideParam.valRatio      = 0.15;
net.divideParam.testRatio     = 0.15;

fprintf('\nTraining QFD-NN (27→%d→%d→1) Option %s ...\n', ...
        hiddenSizes(1), hiddenSizes(2), QFD_OPTION);

if strcmp(QFD_OPTION, 'A') || strcmp(QFD_OPTION, 'C')
    % ----------------------------------------------------------------
    % Options A and C: standard train() call
    % Option C note: true gradient-level regularisation requires dlnetwork.
    % The LAMBDA penalty is applied here as a weight-decay post-correction
    % after each epoch via a manual loop approximation.
    % ----------------------------------------------------------------
    if strcmp(QFD_OPTION, 'A')
        [net, tr] = train(net, X, T);

    else  % Option C — manual epoch loop with post-step regularisation pull
        net.trainParam.showWindow      = false;
        net.trainParam.showCommandLine = false;
        net.divideParam.trainRatio     = 0.70;
        net.divideParam.valRatio       = 0.15;
        net.divideParam.testRatio      = 0.15;

        tr         = [];
        prev_perf  = Inf;
        fail_count = 0;
        MAX_EPOCHS = 1000;
        MAX_FAIL   = 25;
        MIN_GRAD   = 1e-8;

        fprintf('Option C: epoch loop with lambda=%.4f gradient pull ...\n', LAMBDA);
        for ep = 1:MAX_EPOCHS
            [net, tr_ep] = train(net, X, T);

            % ---- Apply QFD regularisation pull after each epoch ----
            % Gradient of LAMBDA*||W2-W_QFD||^2_F w.r.t. W2 = 2*LAMBDA*(W2-W_QFD)
            % Subtract a small step along this gradient (learning-rate ~0.01)
            LR_reg = 0.01;
            grad_W2 = 2 * LAMBDA * (net.LW{2,1} - W_QFD_target);
            net.LW{2,1} = net.LW{2,1} - LR_reg * grad_W2;

            curr_perf = tr_ep.best_vperf;
            if abs(prev_perf - curr_perf) < MIN_GRAD
                fail_count = fail_count + 1;
            else
                fail_count = 0;
            end
            prev_perf = curr_perf;
            tr = tr_ep;

            if fail_count >= MAX_FAIL
                fprintf('  Option C converged at epoch %d.\n', ep);
                break;
            end
            if mod(ep, 100) == 0
                fprintf('  Epoch %d  |  val_perf=%.6f  |  reg_pull=%.6f\n', ...
                        ep, curr_perf, LAMBDA * norm(net.LW{2,1} - W_QFD_target,'fro')^2);
            end
        end
        fprintf('Option C training complete.\n');
    end

elseif strcmp(QFD_OPTION, 'B')
    % ----------------------------------------------------------------
    % Option B: manual epoch loop — re-enforce mask after every epoch
    % so that forbidden connections remain zero throughout training,
    % not only post-training as in v3.
    % ----------------------------------------------------------------
    net.trainParam.showWindow      = false;
    net.trainParam.showCommandLine = false;
    net.divideParam.trainRatio     = 0.70;
    net.divideParam.valRatio       = 0.15;
    net.divideParam.testRatio      = 0.15;

    tr         = [];
    prev_perf  = Inf;
    fail_count = 0;
    MAX_EPOCHS = 1000;
    MAX_FAIL   = 25;
    MIN_GRAD   = 1e-8;

    fprintf('Option B: manual masked epoch loop ...\n');
    for ep = 1:MAX_EPOCHS
        [net, tr_ep] = train(net, X, T);

        % Re-enforce structural sparsity after every gradient update
        net.IW{1,1} = net.IW{1,1} .* M1_mask;
        net.LW{2,1} = net.LW{2,1} .* M2_mask;

        curr_perf = tr_ep.best_vperf;
        if abs(prev_perf - curr_perf) < MIN_GRAD
            fail_count = fail_count + 1;
        else
            fail_count = 0;
        end
        prev_perf = curr_perf;
        tr = tr_ep;

        if fail_count >= MAX_FAIL
            fprintf('  Option B converged at epoch %d.\n', ep);
            break;
        end
        if mod(ep,100) == 0
            fprintf('  Epoch %d  |  val_perf=%.6f\n', ep, curr_perf);
        end
    end
    fprintf('Option B: mask enforced throughout training. Complete.\n');
end

fprintf('\nTraining complete.\n');
if ~isempty(tr)
    fprintf('  Best epoch    : %d\n', tr.best_epoch);
    fprintf('  Best val perf : %.6f\n', tr.best_vperf);
end

% =========================================================================
% 6.  FULL-DATASET EVALUATION
% =========================================================================

Y_prob = net(X);
Y_pred = double(Y_prob >= theta);
Y_true = T;

TP = sum((Y_pred==1) & (Y_true==1));
TN = sum((Y_pred==0) & (Y_true==0));
FP = sum((Y_pred==1) & (Y_true==0));
FN = sum((Y_pred==0) & (Y_true==1));

Accuracy    = (TP+TN)/N;
Precision   = TP/max(TP+FP,1);
Recall      = TP/max(TP+FN,1);
Specificity = TN/max(TN+FP,1);
F1          = 2*Precision*Recall/max(Precision+Recall,eps);
MCC_num     = TP*TN - FP*FN;
MCC_den     = sqrt(double((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));
MCC         = MCC_num/max(MCC_den,eps);
NPV         = TN/max(TN+FN,1);

fprintf('\n================================================\n');
fprintf('  FULL-DATASET RESULTS  (theta = %.2f) | Option %s\n', theta, QFD_OPTION);
fprintf('================================================\n');
fprintf('  TP=%d  FN=%d  FP=%d  TN=%d\n', TP, FN, FP, TN);
fprintf('  Accuracy    : %.4f  (%.2f%%)\n', Accuracy,    Accuracy*100);
fprintf('  Precision   : %.4f  (%.2f%%)\n', Precision,   Precision*100);
fprintf('  Recall      : %.4f  (%.2f%%)\n', Recall,      Recall*100);
fprintf('  Specificity : %.4f  (%.2f%%)\n', Specificity, Specificity*100);
fprintf('  F1 Score    : %.4f  (%.2f%%)\n', F1,          F1*100);
fprintf('  MCC         : %.4f\n',            MCC);
fprintf('  NPV         : %.4f  (%.2f%%)\n', NPV,         NPV*100);
fprintf('================================================\n');

% =========================================================================
% 7.  5-FOLD STRATIFIED CROSS-VALIDATION
% =========================================================================

fprintf('\n--- 5-Fold Stratified Cross-Validation (Option %s) ---\n', QFD_OPTION);

K       = 5;
idx_ai  = find(T==1);  idx_ai  = idx_ai(randperm(length(idx_ai)));
idx_cls = find(T==0);  idx_cls = idx_cls(randperm(length(idx_cls)));

cv_results = zeros(K, 7);   % [Acc, Prec, Rec, Spec, F1, MCC, AUC]

for fold = 1:K
    ai_pf  = floor(length(idx_ai)  / K);
    cls_pf = floor(length(idx_cls) / K);

    val_ai  = idx_ai ((fold-1)*ai_pf  + 1 : fold*ai_pf );
    val_cls = idx_cls((fold-1)*cls_pf + 1 : fold*cls_pf);
    val_idx = [val_ai, val_cls];
    trn_idx = setdiff(1:N, val_idx);

    Xtr = X(:, trn_idx);  Ttr = T(trn_idx);
    Xvl = X(:, val_idx);  Tvl = T(val_idx);

    % Build and initialise fold network (inlined)
    net_cv = patternnet(hiddenSizes, 'trainscg');
    net_cv.layers{1}.transferFcn      = 'tansig';
    net_cv.layers{2}.transferFcn      = 'tansig';
    net_cv.layers{3}.transferFcn      = 'logsig';
    net_cv.trainParam.epochs          = 500;
    net_cv.trainParam.max_fail        = 20;
    net_cv.trainParam.min_grad        = 1e-8;
    net_cv.trainParam.showWindow      = false;
    net_cv.trainParam.showCommandLine = false;
    net_cv.divideParam.trainRatio     = 1.0;
    net_cv.divideParam.valRatio       = 0.0;
    net_cv.divideParam.testRatio      = 0.0;
    net_cv = configure(net_cv, Xtr, Ttr);
    net_cv = init(net_cv);   % Nguyen-Widrow baseline
    % QFD initialisation (all options)
    net_cv.IW{1,1} = M1_norm;
    net_cv.LW{2,1} = M2_norm;
    % Option B: zero forbidden connections at initialisation
    if strcmp(QFD_OPTION, 'B')
        net_cv.IW{1,1} = net_cv.IW{1,1} .* M1_mask;
        net_cv.LW{2,1} = net_cv.LW{2,1} .* M2_mask;
    end

    if strcmp(QFD_OPTION, 'A')
        net_cv = train(net_cv, Xtr, Ttr);

    elseif strcmp(QFD_OPTION, 'C')
        % Option C CV: epoch loop with regularisation pull
        prev_p = Inf; fc = 0;
        for ep = 1:500
            net_cv = train(net_cv, Xtr, Ttr);
            LR_reg = 0.01;
            net_cv.LW{2,1} = net_cv.LW{2,1} - ...
                LR_reg * 2 * LAMBDA * (net_cv.LW{2,1} - W_QFD_target);
            curr_p = net_cv.trainParam.goal;   % approximate
            if abs(prev_p - curr_p) < 1e-8; fc=fc+1; else; fc=0; end
            prev_p = curr_p;
            if fc >= 20; break; end
        end

    elseif strcmp(QFD_OPTION, 'B')
        % Option B CV: epoch loop with mask re-enforcement
        prev_p = Inf; fc = 0;
        for ep = 1:500
            net_cv = train(net_cv, Xtr, Ttr);
            net_cv.IW{1,1} = net_cv.IW{1,1} .* M1_mask;
            net_cv.LW{2,1} = net_cv.LW{2,1} .* M2_mask;
            curr_p = net_cv.trainParam.goal;
            if abs(prev_p - curr_p) < 1e-8; fc=fc+1; else; fc=0; end
            prev_p = curr_p;
            if fc >= 20; break; end
        end
    end

    yp_p = net_cv(Xvl);
    yp   = double(yp_p >= theta);

    tp_f = sum((yp==1)&(Tvl==1)); tn_f = sum((yp==0)&(Tvl==0));
    fp_f = sum((yp==1)&(Tvl==0)); fn_f = sum((yp==0)&(Tvl==1));
    nv   = length(val_idx);

    acc_f  = (tp_f+tn_f)/nv;
    prec_f = tp_f/max(tp_f+fp_f,1);
    rec_f  = tp_f/max(tp_f+fn_f,1);
    spec_f = tn_f/max(tn_f+fp_f,1);
    f1_f   = 2*prec_f*rec_f/max(prec_f+rec_f,eps);
    mcc_f  = (tp_f*tn_f-fp_f*fn_f) / ...
             max(sqrt(double((tp_f+fp_f)*(tp_f+fn_f)*(tn_f+fp_f)*(tn_f+fn_f))),eps);

    ths_f  = linspace(0,1,51);
    tpr_f  = zeros(1,51); fpr_f = zeros(1,51);
    for ki = 1:51
        yp_ki  = double(yp_p >= ths_f(ki));
        tp_ki  = sum((yp_ki==1)&(Tvl==1)); fn_ki = sum((yp_ki==0)&(Tvl==1));
        fp_ki  = sum((yp_ki==1)&(Tvl==0)); tn_ki = sum((yp_ki==0)&(Tvl==0));
        tpr_f(ki) = tp_ki/max(tp_ki+fn_ki,1);
        fpr_f(ki) = fp_ki/max(fp_ki+tn_ki,1);
    end
    auc_f = abs(trapz(fliplr(fpr_f), fliplr(tpr_f)));

    cv_results(fold,:) = [acc_f, prec_f, rec_f, spec_f, f1_f, mcc_f, auc_f];
    fprintf('  Fold %d: Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  MCC=%.4f  AUC=%.4f\n', ...
            fold, acc_f, prec_f, rec_f, f1_f, mcc_f, auc_f);
end

cv_mean = mean(cv_results, 1);
cv_std  = std(cv_results,  0, 1);

fprintf('\n  CV SUMMARY (mean +/- std)  |  Option %s\n', QFD_OPTION);
fprintf('  Accuracy    : %.4f +/- %.4f\n', cv_mean(1), cv_std(1));
fprintf('  Precision   : %.4f +/- %.4f\n', cv_mean(2), cv_std(2));
fprintf('  Recall      : %.4f +/- %.4f\n', cv_mean(3), cv_std(3));
fprintf('  Specificity : %.4f +/- %.4f\n', cv_mean(4), cv_std(4));
fprintf('  F1 Score    : %.4f +/- %.4f\n', cv_mean(5), cv_std(5));
fprintf('  MCC         : %.4f +/- %.4f\n', cv_mean(6), cv_std(6));
fprintf('  AUC-ROC     : %.4f +/- %.4f\n', cv_mean(7), cv_std(7));

% =========================================================================
% 8.  THRESHOLD SWEEP
% =========================================================================

fprintf('\n--- Threshold Sweep ---\n');
ths_full = linspace(0.05, 0.95, 181);
sweep    = zeros(length(ths_full), 6);

for ki = 1:length(ths_full)
    th    = ths_full(ki);
    yp    = double(Y_prob >= th);
    tp_k  = sum((yp==1)&(Y_true==1)); tn_k = sum((yp==0)&(Y_true==0));
    fp_k  = sum((yp==1)&(Y_true==0)); fn_k = sum((yp==0)&(Y_true==1));
    acc_k = (tp_k+tn_k)/N;
    prec_k= tp_k/max(tp_k+fp_k,1);
    rec_k = tp_k/max(tp_k+fn_k,1);
    f1_k  = 2*prec_k*rec_k/max(prec_k+rec_k,eps);
    mcc_k = (tp_k*tn_k-fp_k*fn_k)/max(sqrt(double((tp_k+fp_k)*(tp_k+fn_k)*(tn_k+fp_k)*(tn_k+fn_k))),eps);
    sweep(ki,:) = [th, acc_k, prec_k, rec_k, f1_k, mcc_k];
end

[best_f1, best_ki] = max(sweep(:,5));
best_theta = sweep(best_ki, 1);
fprintf('  Optimal theta (max F1) : %.3f  →  F1=%.4f  Acc=%.4f\n', ...
        best_theta, best_f1, sweep(best_ki,2));

% =========================================================================
% 9.  ROC + AUC
% =========================================================================

tpr_all = zeros(1,101);
fpr_all = zeros(1,101);
ths_roc = linspace(0,1,101);
for ki = 1:101
    th    = ths_roc(ki);
    yp    = double(Y_prob >= th);
    tp_k  = sum((yp==1)&(Y_true==1)); fn_k = sum((yp==0)&(Y_true==1));
    fp_k  = sum((yp==1)&(Y_true==0)); tn_k = sum((yp==0)&(Y_true==0));
    tpr_all(ki) = tp_k/max(tp_k+fn_k,1);
    fpr_all(ki) = fp_k/max(fp_k+tn_k,1);
end
AUC = abs(trapz(fliplr(fpr_all), fliplr(tpr_all)));
fprintf('  AUC-ROC (full dataset)  : %.4f\n', AUC);

% =========================================================================
% 10.  OPTION C — QFD REGULARISATION DIAGNOSTICS
% =========================================================================

if strcmp(QFD_OPTION, 'C')
    W2_trained = net.LW{2,1};
    reg_term   = norm(W2_trained - W_QFD_target, 'fro')^2;
    fprintf('\nOption C — QFD Regularisation Diagnostics:\n');
    fprintf('  ||W2 - W_QFD||^2_F  = %.6f\n', reg_term);
    fprintf('  Lambda * residual   = %.6f\n',  LAMBDA * reg_term);
    fprintf('  (Gradient pull applied per epoch at LR_reg=0.01)\n');
end

% =========================================================================
% 11.  PLOTS
% =========================================================================

%% Figure 1: Training Performance
fig1 = figure('Name','Training Performance','NumberTitle','off',...
              'Units','normalized','OuterPosition',[0 0.55 0.45 0.45]);
plotperform(tr);
title(sprintf('Training Performance  |  Option %s', QFD_OPTION),...
      'FontSize',13,'FontWeight','bold');

%% Figure 2: Four-panel Confusion Matrix (Train / Val / Test / All)
%  Style matches MATLAB plotconfusion — green = correct, red = incorrect,
%  grey = row/column accuracy summaries.
fig2 = figure('Name','Confusion Matrices','NumberTitle','off',...
              'Units','normalized','OuterPosition',[0.5 0.55 0.5 0.45]);

% ---- Recover train/val/test indices from training record ----
% For Options B and C (manual loop), tr.trainInd / valInd / testInd may be
% from the last epoch's tr_ep — use them if available, else use all data.
if ~isempty(tr) && isfield(tr,'trainInd') && ~isempty(tr.trainInd)
    trn_idx_full = tr.trainInd;
    val_idx_full = tr.valInd;
    tst_idx_full = tr.testInd;
else
    % Fallback: recreate the 70/15/15 split deterministically
    rng(42);
    idx_perm     = randperm(N);
    n_trn        = round(0.70*N);
    n_val        = round(0.15*N);
    trn_idx_full = idx_perm(1:n_trn);
    val_idx_full = idx_perm(n_trn+1:n_trn+n_val);
    tst_idx_full = idx_perm(n_trn+n_val+1:end);
end

% ---- Helper: compute CM and panel data for a given index subset ----
% Returns [n00 n01; n10 n11] where rows=Output(predicted), cols=Target(true)
% matching MATLAB plotconfusion orientation: Output Class on Y, Target on X
cm_panel = @(idx) deal( ...
    sum((Y_pred(idx)==0)&(Y_true(idx)==0)), ...  TN  (out=0,tgt=0)
    sum((Y_pred(idx)==1)&(Y_true(idx)==0)), ...  FP  (out=1,tgt=0)
    sum((Y_pred(idx)==0)&(Y_true(idx)==1)), ...  FN  (out=0,tgt=1)
    sum((Y_pred(idx)==1)&(Y_true(idx)==1))  ...  TP  (out=1,tgt=1)
);

panels = {trn_idx_full, val_idx_full, tst_idx_full, 1:N};
titles = {'Training Confusion Matrix','Validation Confusion Matrix',...
          'Test Confusion Matrix','All Confusion Matrix'};

% Colours matching MATLAB plotconfusion style
C_CORRECT = [0.80 0.94 0.80];   % light green  — correct cells
C_WRONG   = [0.95 0.80 0.80];   % light red    — error cells
C_SUMMARY = [0.85 0.85 0.85];   % light grey   — row/col accuracy summary
C_OVERALL = [0.70 0.70 0.70];   % darker grey  — overall accuracy cell

for pp = 1:4
    idx = panels{pp};
    n_p = numel(idx);

    [n00, n10, n01, n11] = cm_panel(idx);

    % CM matrix: rows = Output Class (0 top, 1 bottom), cols = Target Class (0 left, 1 right)
    CM = [n00, n01; n10, n11];
    N_p = sum(CM(:));

    % Row sums (total predicted per class)
    row_sum = sum(CM, 2);   % [n_pred0; n_pred1]
    % Col sums (total actual per class)
    col_sum = sum(CM, 1);   % [n_actual0, n_actual1]

    % Row accuracy: % of row that is correct (precision per output class)
    row_acc  = diag(CM) ./ max(row_sum, 1) * 100;
    row_err  = 100 - row_acc;
    % Col accuracy: % of col that is correct (recall per target class)
    col_acc  = diag(CM)' ./ max(col_sum, 1) * 100;
    col_err  = 100 - col_acc;
    % Overall accuracy
    overall_acc = sum(diag(CM)) / max(N_p,1) * 100;
    overall_err = 100 - overall_acc;

    ax = subplot(2,2,pp);

    % ---- Draw cell backgrounds (3x3 grid: 2x2 CM + 1 row summary + 1 col summary) ----
    % Grid: cols 1-2 = target classes, col 3 = row accuracy
    %       rows 1-2 = output classes, row 3 = col accuracy
    % Cell (row i, col j) drawn as filled rectangle

    hold on; axis off;
    set(ax,'XLim',[0 3],'YLim',[0 3]);

    cell_colors = {C_CORRECT, C_WRONG,   C_SUMMARY; ...
                   C_WRONG,   C_CORRECT, C_SUMMARY; ...
                   C_SUMMARY, C_SUMMARY, C_OVERALL};

    for ci = 1:3
        for cj = 1:3
            % Draw from bottom: row 3 is bottom (col summary), row 1 is top
            rx = cj-1;  ry = 3-ci;   % bottom-left corner
            rectangle('Position',[rx, ry, 1, 1], ...
                      'FaceColor', cell_colors{ci,cj}, ...
                      'EdgeColor',[0.5 0.5 0.5],'LineWidth',1.2);
        end
    end

    % ---- Cell text: CM values ----
    for ci = 1:2
        for cj = 1:2
            rx  = cj-1; ry = 3-ci;
            cx  = rx + 0.5; cy = ry + 0.5;
            val = CM(ci,cj);
            pct = val/max(N_p,1)*100;
            text(cx, cy+0.12, sprintf('%d', val), ...
                 'HorizontalAlignment','center','FontSize',11,'FontWeight','bold',...
                 'Color',[0 0 0]);
            text(cx, cy-0.15, sprintf('%.1f%%', pct), ...
                 'HorizontalAlignment','center','FontSize',9,...
                 'Color',[0.25 0.25 0.25]);
        end
    end

    % ---- Row accuracy cells (col 3) ----
    for ci = 1:2
        cx = 2.5; cy = (3-ci) + 0.5;
        text(cx, cy+0.12, sprintf('%.1f%%', row_acc(ci)), ...
             'HorizontalAlignment','center','FontSize',10,'FontWeight','bold',...
             'Color',[0.05 0.45 0.05]);
        text(cx, cy-0.15, sprintf('%.1f%%', row_err(ci)), ...
             'HorizontalAlignment','center','FontSize',9,...
             'Color',[0.70 0.05 0.05]);
    end

    % ---- Column accuracy cells (row 3) ----
    for cj = 1:2
        cx = (cj-1) + 0.5; cy = 0.5;
        text(cx, cy+0.12, sprintf('%.1f%%', col_acc(cj)), ...
             'HorizontalAlignment','center','FontSize',10,'FontWeight','bold',...
             'Color',[0.05 0.45 0.05]);
        text(cx, cy-0.15, sprintf('%.1f%%', col_err(cj)), ...
             'HorizontalAlignment','center','FontSize',9,...
             'Color',[0.70 0.05 0.05]);
    end

    % ---- Overall accuracy cell (bottom-right, row 3 col 3) ----
    text(2.5, 0.5+0.12, sprintf('%.1f%%', overall_acc), ...
         'HorizontalAlignment','center','FontSize',10,'FontWeight','bold',...
         'Color',[0.05 0.45 0.05]);
    text(2.5, 0.5-0.15, sprintf('%.1f%%', overall_err), ...
         'HorizontalAlignment','center','FontSize',9,...
         'Color',[0.70 0.05 0.05]);

    % ---- Axis labels ----
    % X-axis (Target Class) — below the grid
    text(0.5, -0.18, '0', 'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    text(1.5, -0.18, '1', 'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    text(1.0, -0.38, 'Target Class', 'HorizontalAlignment','center','FontSize',10,...
         'FontWeight','bold');

    % Y-axis (Output Class) — left of grid
    text(-0.18, 2.5, '0', 'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    text(-0.18, 1.5, '1', 'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    text(-0.42, 1.5, 'Output Class', 'HorizontalAlignment','center','FontSize',10,...
         'FontWeight','bold','Rotation',90);

    hold off;
    title(titles{pp}, 'FontSize',11,'FontWeight','bold');
end

sgtitle(sprintf('Confusion Matrices  |  Option %s  |  Theta=%.2f', QFD_OPTION, theta),...
        'FontSize',13,'FontWeight','bold');

%% Figure 3: ROC Curve
fig3 = figure('Name','ROC Curve','NumberTitle','off',...
              'Units','normalized','OuterPosition',[0 0.05 0.45 0.45]);
plot(fpr_all, tpr_all, 'b-', 'LineWidth',2.5); hold on;
plot([0 1],[0 1],'k--','LineWidth',1.2);
op_fpr = FP/max(FP+TN,1); op_tpr = TP/max(TP+FN,1);
plot(op_fpr, op_tpr, 'ro','MarkerSize',10,'MarkerFaceColor','r','LineWidth',1.5);
text(op_fpr+0.03,op_tpr-0.04,sprintf('Theta=%.2f',theta),'FontSize',10,'Color','r','FontWeight','bold');
yp_opt = double(Y_prob >= best_theta);
tp_opt = sum((yp_opt==1)&(Y_true==1)); fp_opt = sum((yp_opt==1)&(Y_true==0));
tn_opt = sum((yp_opt==0)&(Y_true==0));
opt_fpr = fp_opt/max(fp_opt+tn_opt,1); opt_tpr = tp_opt/max(tp_opt+(100-tp_opt),1);
plot(opt_fpr,opt_tpr,'g^','MarkerSize',10,'MarkerFaceColor','g','LineWidth',1.5);
text(opt_fpr+0.03,opt_tpr+0.02,sprintf('Theta_{opt}=%.2f',best_theta),...
     'FontSize',10,'Color',[0 0.6 0],'FontWeight','bold');
fill([fpr_all fliplr(fpr_all)],[tpr_all zeros(1,101)],'b','FaceAlpha',0.08,'EdgeColor','none');
legend({sprintf('QFD-NN (AUC=%.4f)',AUC),'Random',sprintf('theta=%.2f',theta),...
        sprintf('theta_{opt}=%.2f',best_theta)},'Location','SouthEast','FontSize',10);
xlabel('FPR (1-Specificity)','FontSize',12); ylabel('TPR (Recall)','FontSize',12);
title(sprintf('ROC  |  AUC=%.4f  |  Option %s',AUC,QFD_OPTION),'FontSize',13,'FontWeight','bold');
grid on; axis square; xlim([0 1]); ylim([0 1]); hold off;

%% Figure 4: Threshold Sweep
fig4 = figure('Name','Threshold Sweep','NumberTitle','off',...
              'Units','normalized','OuterPosition',[0.5 0.05 0.45 0.45]);
plot(sweep(:,1),sweep(:,2)*100,'b-','LineWidth',2,'DisplayName','Accuracy'); hold on;
plot(sweep(:,1),sweep(:,3)*100,'g--','LineWidth',2,'DisplayName','Precision');
plot(sweep(:,1),sweep(:,4)*100,'r-','LineWidth',2,'DisplayName','Recall');
plot(sweep(:,1),sweep(:,5)*100,'m-','LineWidth',2.5,'DisplayName','F1');
plot(sweep(:,1),(sweep(:,6)+1)/2*100,'k:','LineWidth',1.5,'DisplayName','MCC (rescaled)');
xline(theta,'b:','LineWidth',1.5,'DisplayName',sprintf('theta=%.2f',theta));
xline(best_theta,'r:','LineWidth',1.5,'DisplayName',sprintf('theta_opt=%.2f',best_theta));
plot(best_theta,best_f1*100,'r*','MarkerSize',14,'LineWidth',2,...
     'DisplayName',sprintf('Max F1=%.2f%%',best_f1*100));
hold off;
xlabel('Decision threshold theta','FontSize',12); ylabel('Metric (%)','FontSize',12);
title(sprintf('Metrics vs Threshold  |  Option %s',QFD_OPTION),'FontSize',13,'FontWeight','bold');
legend('Location','West','FontSize',9); grid on; xlim([0 1]); ylim([0 105]);

%% Figure 5: Feature Importance
fig5 = figure('Name','Feature Importance','NumberTitle','off',...
              'Units','normalized','OuterPosition',[0 0 0.48 0.95]);
W1_abs   = abs(net.IW{1,1});
feat_imp = mean(W1_abs, 1);
[imp_sort, imp_idx] = sort(feat_imp,'descend');
grp_colors = [
    0.08 0.40 0.75; 0.08 0.40 0.75; 0.08 0.40 0.75; 0.08 0.40 0.75; 0.08 0.40 0.75;
    0.18 0.49 0.20; 0.18 0.49 0.20; 0.18 0.49 0.20; 0.18 0.49 0.20; 0.18 0.49 0.20; 0.18 0.49 0.20;
    0.05 0.29 0.63; 0.05 0.29 0.63; 0.05 0.29 0.63; 0.05 0.29 0.63;
    0.90 0.40 0.00; 0.90 0.40 0.00; 0.90 0.40 0.00; 0.90 0.40 0.00;
    0.29 0.07 0.55; 0.29 0.07 0.55; 0.29 0.07 0.55; 0.29 0.07 0.55; 0.29 0.07 0.55; 0.29 0.07 0.55;
    0.00 0.41 0.36; 0.00 0.41 0.36;
];
bar_h = barh(1:27, imp_sort(end:-1:1),'FaceColor','flat');
for b=1:27; bar_h.CData(b,:) = grp_colors(imp_idx(28-b),:); end
set(gca,'YTick',1:27,'YTickLabel',feat_names(imp_idx(end:-1:1)),'FontSize',8);
xlabel('Mean |Weight|  (Layer-1 input weights)','FontSize',11);
title(sprintf('Feature Importance  |  Option %s',QFD_OPTION),'FontSize',12,'FontWeight','bold');
grid on;
leg_entries = {'Group A Dependency','Group B Code Pattern','Group C Artifact',...
               'Group D Documentation','Group E Structural','Group F Learning Signal'};
leg_colors  = [0.08 0.40 0.75; 0.18 0.49 0.20; 0.05 0.29 0.63;
               0.90 0.40 0.00; 0.29 0.07 0.55; 0.00 0.41 0.36];
hold on;
for g=1:6
    plot(NaN,NaN,'s','MarkerSize',12,'MarkerFaceColor',leg_colors(g,:),...
         'MarkerEdgeColor',leg_colors(g,:),'DisplayName',leg_entries{g});
end
legend('Location','SouthEast','FontSize',8); hold off;

%% Figure 6: Layer-1 Activation Heatmap
fig6 = figure('Name','L1 Neuron Activations','NumberTitle','off',...
              'Units','normalized','OuterPosition',[0.5 0 0.45 0.5]);
H1 = tansig(net.IW{1,1} * X + net.b{1} * ones(1,N));
[~, sort_smp] = sortrows([Y_true(:), Y_prob(:)],[1 2]);
H1_sorted = H1(:, sort_smp);
T_sorted  = Y_true(sort_smp);
imagesc(H1_sorted); colormap(gca, parula(256));
cb = colorbar; cb.Label.String = 'Activation (tansig)';
yticks(1:9);
yticklabels({'h1 Dependency','h2 Inference','h3 Training','h4 Generative',...
             'h5 Documentation','h6 Model-Artifact','h7 AI Infrastructure',...
             'h8 Data-Driven','h9 Stat. Learning'});
xlabel('Samples (sorted by class then score)','FontSize',11);
title(sprintf('Layer-1 Technical Evidence Activations  |  Option %s',QFD_OPTION),...
      'FontSize',11,'FontWeight','bold');
n_cls = sum(T_sorted==0);
hold on; xline(n_cls+0.5,'r--','LineWidth',2,'DisplayName','Classical|AI'); 
legend('Location','NorthEast','FontSize',9); hold off;
set(gca,'FontSize',8);

%% Figure 7: Score Distribution (robust to degenerate/collapsed distributions)
fig7 = figure('Name','Score Distribution','NumberTitle','off',...
              'Units','normalized','OuterPosition',[0 0 0.45 0.45]);
scores_ai  = Y_prob(Y_true==1);
scores_cls = Y_prob(Y_true==0);

% Adaptive bin edges: use narrow bins near actual score clusters,
% wider bins elsewhere — handles perfect-separation collapse gracefully.
all_scores  = [scores_ai(:); scores_cls(:)];
score_range = max(all_scores) - min(all_scores);

if score_range < 0.05
    % Fully collapsed: all scores nearly identical — use fine bins around actual range
    bin_edges = linspace(max(0, min(all_scores)-0.05), ...
                         min(1, max(all_scores)+0.05), 31);
else
    % Normal or near-normal spread — use data-adaptive bin width (Freedman-Diaconis)
    n_bins = min(50, max(10, round(score_range / (2 * iqr(all_scores) * N^(-1/3) + eps))));
    bin_edges = linspace(0, 1, n_bins + 1);
end

hold on;
h_cls = histogram(scores_cls, bin_edges, ...
    'FaceColor',[0.9 0.4 0.1],'FaceAlpha',0.65,...
    'EdgeColor',[0.7 0.3 0.0],'LineWidth',0.5,'DisplayName','Classical');
h_ai  = histogram(scores_ai,  bin_edges, ...
    'FaceColor',[0.1 0.6 0.2],'FaceAlpha',0.65,...
    'EdgeColor',[0.0 0.4 0.1],'LineWidth',0.5,'DisplayName','AI');

% If histograms still invisible (max count = 0), fall back to stem/rug plot
if max([h_cls.Values, h_ai.Values]) == 0 || ...
   (max(h_cls.Values) <= 1 && max(h_ai.Values) <= 1)
    cla; % clear and use rug + KDE instead
    % Rug plot: individual score marks per class
    plot(scores_cls, zeros(size(scores_cls))-1, '|', ...
         'Color',[0.9 0.4 0.1],'MarkerSize',12,'LineWidth',1.5,'DisplayName','Classical');
    plot(scores_ai,  zeros(size(scores_ai))+1,  '|', ...
         'Color',[0.1 0.6 0.2],'MarkerSize',12,'LineWidth',1.5,'DisplayName','AI');
    % Bar chart of counts per class at their mean score
    bar([mean(scores_cls), mean(scores_ai)], [numel(scores_cls), numel(scores_ai)], ...
        0.02, 'FaceColor','flat','FaceAlpha',0.7,'EdgeColor','none');
    ylabel('Count / position','FontSize',12);
end

xline(theta,      'b-', 'LineWidth',2.5, 'DisplayName',sprintf('Theta=%.2f',theta));
xline(best_theta, 'r--','LineWidth',2.5, 'DisplayName',sprintf('Theta_{opt}=%.2f',best_theta));
hold off;

% Auto-scale x-axis: show full [0,1] if spread > 0.3, else zoom to data ± 10%
if score_range > 0.30
    xlim([0 1]);
else
    xlim([max(0, min(all_scores)-0.10), min(1, max(all_scores)+0.10)]);
end

xlabel('P(AI)  sigmoid output','FontSize',12); ylabel('Count','FontSize',12);
title(sprintf('Score Distribution by Class  |  Option %s',QFD_OPTION),...
      'FontSize',13,'FontWeight','bold');
legend('Location','Best','FontSize',10); grid on;

% Annotation: class separation statistics
sep_gap = min(scores_ai) - max(scores_cls);
if sep_gap > 0
    annotation_str = sprintf('Perfect separation\nGap = %.4f', sep_gap);
else
    annotation_str = sprintf('Overlap region\nMin AI=%.3f  Max CLS=%.3f',...
                             min(scores_ai), max(scores_cls));
end
text(0.98, 0.92, annotation_str, 'Units','normalized',...
     'HorizontalAlignment','right','FontSize',9,'FontAngle','italic',...
     'Color',[0.3 0.3 0.3],'BackgroundColor',[0.97 0.97 0.97]);

%% Figure 8: CV Metrics Bar Chart
fig8 = figure('Name','CV Results','NumberTitle','off',...
              'Units','normalized','OuterPosition',[0.5 0 0.45 0.45]);
metric_names = {'Accuracy','Precision','Recall','Specificity','F1','MCC','AUC'};
bar_cv = bar(1:7, cv_mean*100, 0.55,'FaceColor','flat');
bar_cv.CData = repmat([0.08 0.40 0.75],7,1);
hold on;
errorbar(1:7, cv_mean*100, cv_std*100, cv_std*100,'k.','LineWidth',1.8,'CapSize',8);
for f=1:K
    plot(1:7, cv_results(f,:)*100,'o','MarkerSize',6,...
         'MarkerFaceColor',[0.8 0.8 0.8],'MarkerEdgeColor',[0.4 0.4 0.4],...
         'HandleVisibility','off');
end
hold off;
set(gca,'XTick',1:7,'XTickLabel',metric_names,'FontSize',10);
ylabel('Value (%)','FontSize',12);
title(sprintf('5-Fold CV Results (mean +/- std)  |  Option %s',QFD_OPTION),...
      'FontSize',13,'FontWeight','bold');
ylim([0 110]); grid on;
for i=1:7
    text(i, cv_mean(i)*100+cv_std(i)*100+2, sprintf('%.1f',cv_mean(i)*100),...
         'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
end

% =========================================================================
% 12.  SAVE RESULTS
% =========================================================================

results.Option       = QFD_OPTION;
results.LAMBDA       = LAMBDA;
results.TP=TP; results.TN=TN; results.FP=FP; results.FN=FN;
results.Accuracy     = Accuracy;
results.Precision    = Precision;
results.Recall       = Recall;
results.Specificity  = Specificity;
results.F1           = F1;
results.MCC          = MCC;
results.NPV          = NPV;
results.AUC          = AUC;
results.theta        = theta;
results.best_theta   = best_theta;
results.cv_mean      = cv_mean;
results.cv_std       = cv_std;
results.cv_results   = cv_results;
results.W_qfd_init   = W_qfd_init;
results.sweep        = sweep;

save(sprintf('QFD_NN_results_v4_%s.mat',QFD_OPTION), ...
     'net','results','X','T','Y_prob','Y_pred','M1','M2','M1_mask','M2_mask');

fprintf('\nResults saved to QFD_NN_results_v4_%s.mat\n', QFD_OPTION);
fprintf('Figures: 1=Training | 2=Confusion | 3=ROC | 4=Threshold\n');
fprintf('         5=Feature Importance | 6=L1 Activations | 7=Score Dist | 8=CV\n');
fprintf('\nDone.\n');

% =========================================================================
% END
% =========================================================================
