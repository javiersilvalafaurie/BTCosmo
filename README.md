# BTCosmo

The BTCosmo (Bayesian and covariance Tapering in Cosmology) includes the codes used in https://arxiv.org/abs/1912.06155 and <<>> to compute the corrected precision matrix using Hartlap correction and covariance tapering, both detailed in Paz et al. (Paz D. J., Sánchez A. G., 2015, MNRAS, 454, 4, 4326). Together with and MCMC algorithm (based in https://emcee.readthedocs.io/en/stable/) to constrain the cosmological information in Galaxy Clustering through the two-point correlation function.

The codes are in jupyter notebooks and python files. To run them we recommend Anaconda, which includes all the libraries. Nevertheless you also need to install CLASS library (https://github.com/lesgourg/class_public).

In xi_0_2_power_spectrum_creator.ipynb we created the power spectrums with and without wiggles using a fiducial cosmology (Planck 2015).

In xi_0_2_MCMC.ipynb we run the MCMC algorithm to constrain the cosmological information. 

These codes belong to Javier Silva Lafaurie and can be used freely under the MIT license. If you are going to use them please cite https://arxiv.org/abs/1912.06155 and/or <<>>. For any question please contact me: javier.silva@ug.uchile.cl
