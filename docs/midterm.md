# Midterm summary

## Project introduction
- What is the project
- What are the goals
- Why does it matter

## Data [W1, W2, W3, W4t6, W7, code]
- Presentation
    - Sources
    - Pipeline (dl, pp, dataset + add prediction and eval [W3, W4t6, code])
        - Already make it clear here that the S2 preprocessing (warping, interpolation) is out of our control
        - Standardization [W4t6, W7, code:eval]
- Analysis
    - Visualize projects (norway map, on QGis)
    - Visualize 1 project(CV stripes, mask, etc...)

## Evaluation [W2, W3, W7, OL, code]
- Definitions (~literature review)
    - Regression
    - Calibration: define calibration (classification) as an agreement between confidence and class probability, redefine for regression (2 directions: UCE/ENCE or AUCE for gaussian)
    - Usability: calibration metrics flaws => ensure usable
- Selected metrics
    - Regression
    - Calibration
    - Usability
- [Opt] Qualitative metrics: maybe we just explain when they show up
- Implementation details [W7, code]

## Experiments

### Baseline [W4t6, W7]
- cleanest dates (365, show date distro again)
- results [W7]

### Cloud Experiment [W12, code]
- Motivation plot [W12s4]
- Settings
    - Data acquisition pipeline (GEE) [code]
        - Warping [W13]
            - GDAL vs rio
            - rio diff
    - Dataset creation differences [code]
    - Evaluation setting [W14]
        - Sanity check
        - GEE
        - [Opt] Cloud detection
- Results [W14]
    - Sanity check: Baseline vs GEE
    - GEE results
    - [Opt] Cloud detection
- Discussion
    - Sanity check => conclusion
    - GEE results => conclusion
    - [Opt] Cloud detection => conclusion

## Next steps
- Next experiments? to define

## Conclusion
- imperfect calibration
- define next steps

## Appendix
- Normality assumption
    - Why:
        - MLE training assumes Gaussian GT and pred, hence residuals too
        - AUCE assumes the residuals are normal
    - Results [W8]
    - Potential solution (See future work added to all slide shows)
- Spatial autocorrelation
    - Why:
        - If there is spatial autocorrelation, we may want to remove this contribution from the prediction to get more reliable metrics
    - Results [W8s9] (Spoiler alert: no)
- Cloud threshold experiment [W9]


