# Bilateral Filter for Contrast Reduction

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

#### Bilateral Filtering

<img src="/images/swamp.jpg" alt="(Left) Original Image (Right) Filtered Image" width="1200" height="318">
<pre>
                  Fig.1 - (Left) Original Image (Right) Filtered Image </pre>

#### Contrast Reduction

<img src="/images/dragon.png" alt="(Left) Original Image (Right) Filtered Image" width="952" height="576">
<pre>
                  Fig.2 - (Left) Original Image (Right) Filtered Image </pre>
                  
<img src="/images/tulip.png" alt="(Left) Original Image (Right) Filtered Image" width="1000" height="480">
<pre>
                  Fig.3 - (Left) Original Image (Right) Filtered Image </pre>

### References

[1] [Frédo Durand, Julie Dorsey - Fast Bilateral Filtering for the Display of High-Dynamic-Range](https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf)
