# Bilateral Filter Contrast Reduction

### HDR Imaging

This project is an implementation of a fast bilateral filter for contrast reduction (which can be used to display HDR images). The technique implemented is discussed in [1]. Broadly the steps involved are:
1. Implementation of a bilateral filter
2. Downsampling in spatial and intensity domains
3. Tone mapping using the bilateral filter by using detail and base components of the image 

### Usage

To compute the response of the bilateral filter run using,
`python run_bf.py --imagepath PATH_TO_IMAGE`

To perform contrast reduction on an image run,
`python run_bf_tonemap.py --imagepath PATH_TO_IMAGE`

### Results

<img src="/images/1_4.png" alt="Original Image (Low Exposure)" width="337" height="445">     <img src="/images/32_1.png" alt="Original Image (Low Exposure)" width="337" height="445">
<pre>
                  Fig.1 - Original Images (Left - Low Exposure Image, Right - High Exposure Image) </pre>

<img src="/images/0_Calib_Chapel_CRF0.jpg" alt="Global Tonemapped HDR Image" width="337" height="445">     <img src="/images/0_Calib_Chapel_local_CRF0.jpg" alt="Local Tonemapped HDR Image" width="337" height="445">

<pre>
                  Fig.2 - HDR Images (Left - Global Tonemapping, Right - Local Tonemapping) </pre>

### References

[1] [Frédo Durand, Julie Dorsey - Fast Bilateral Filtering for the Display of High-Dynamic-Range](https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf)
