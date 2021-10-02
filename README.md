## Learned Holographic Light Transport
[Koray Kavaklı](https://www.linkedin.com/in/koray-kavakli-75949241/),
[Hakan Ürey](https://mems.ku.edu.tr/),
and [Kaan Akşit](https://kaanaksit.com)

<img src='https://github.com/kunguz/realistic_holography/raw/main/result.gif' width=640>


[[Manuscript]](https://arxiv.org/abs/2108.08253), [[Dataset]](https://doi.org/10.5522/04/15087867.v1)

# Description
This work introduces a learned method to improve image reconstructions in an actual phase-only holographic display.  
The technical details of the work are detailed in our [manuscript](https://arxiv.org/pdf/2108.08253). 
Therefore, we will skip describing the work and help you to get this codebase running at your end.

## Quickstart

### (0) Install the required packages
To run our code, you have to install the required packages. 
The latest and greatest version of `odak` can be installed using the following syntax in a Unix shell:

```
pip3 install odak
```

Note that there is a setting file located in `settings/sample.txt`. 
This file can help set variables such as input file locations, desired output locations, or physical simulation qualities.

### (1) Running the code with a pretrained model
We supply a [pretrained model](calibrations/kernel.pt) within this repository. 
This model is valid for our holographic display. 
In any means, we ask you to have the first run with this pre-trained model by running:

```
python3 main.py
```

As the run is completed successfully, you know that you have the dependencies installed correctly and a reliable codebase.
After this run, you will find a new directory called `output`. 
You will find phase-only holograms tailored for our Spatial Light Modulator (SLM) within this new directory. 

### (2) Training for your holographic display
Naturally, you would want to have a model for your holographic display so that you can improve visual quality accordingly. 
In our case, we train for our holographic display and provide the [dataset](https://doi.org/10.5522/04/15087867.v1) from our training. 
In our training dataset, you will find optimized holograms and homography corrected photographs captured from our display. Homography correction gets the images aligned with the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) so that we can compare them against a simulation.

Note that we do not provide you toolkit to generate holograms with ideal models, capture images and correct homography. 
Curious folks can consult the documentation of [`odak`](https://kunguz.github.io/odak/cgh/) to generate holograms using ideal models.
As capturing images or correcting homography are very much dependent on your setup, we skip providing such a toolkit.
You have to prepare such a dataset for your case on your own. 
Once you have such a dataset, all you need is to first remove the existing model by running:

```
rm -Rf calibrations
```

And edit `settings/sample.txt` such that the pixel pitch, distance or SLM resolution is correct.
 For example, a sample settings file looks like below.

```
{
    "general"      : {
                      "cuda"                    : 1,
                      "iterations"              : 400,
                      "propagation type"        : "custom",
                      "output directory"        : "./output",
                      "target filename"         : "./inputs/indian_head.png",
                      "loss weights"            : [1.0],
                      "region of interest"      : [0.0,1.0],
                      "learning rate"           : 0.1
                     },

    "ideal"        : {
                      "multiplier"              : 5.0
                     },

    "kernel"       : {
                      "region of interest"      : [0.1,0.9],
                      "learning rate"           : 0.002,
                      "iterations"              : 30,
                      "multiplier"              : 0.7
                     },

    "dataset"      : {
                      "input directory"         : "../datasets/DIV2K_koray_holography/div2k_holograms",
                      "output directory"        : "../datasets/DIV2K_koray_holography/div2k_warped"
                     },

    "image"        : {
                      "location"                : [0.0,0.0,0.15]
                     },

    "slm"          : {
                      "model"                   : "Holoeye Pluto 2.1",
                      "pixel pitch"             : 0.000008,
                      "resolution"              : [1080,1920]
                     },

    "beam"         : {
                      "wavelength"              : 0.000000515
                     }
}
```

To start the learning process, simply run:

```
python3 main.py
```

Once the learning is complete, you can always visit `settings/sample.txt` and edit the target image to generate holograms both for an ideal case and learned model case. 
The results will appear in the `output` directory.
In some cases, we observe changing the `multiplier` of `kernel` inside `settings/sample.txt` changes the noise pattern in the final image. 
For our experiments, we kept it at `0.7`. 
However, this number may be different for your setup.

### (3) Getting help beyond this description
We will be more than happy to assist your holographic display efforts, and you can always reach us either through the `issues` section or via email (`kaanaksit@kaanaksit.com`).

# Citation
If you find this repository useful for your research, please consider citing our work using the below `BibTeX entry.

```
@inproceedings{koray2021learned,
  title={Learned Holographic Light Transport},
  author={Kavakl{\i}, Koray and Urey, Hakan and Ak\c{s}it, Kaan},
  booktitle={Applied Optics},
  year={2021}
}
```

# Acknowledgements
The authors thank the anonymous reviewers for their helpful feedback. 
The authors also thank [Oliver Kingshott](http://oliver.kingshott.com/) and [Duygu Ceylan](https://www.duygu-ceylan.com/) for the fruitful and inspiring discussions improving the outcome of this research and [Selim Ölçer](https://www.linkedin.com/in/selim-olcer-96413822?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B%2B3nvRGtUR1mExhUCrOrvJg%3D%3D) for helping with the fibre alignment of laser light source in the proof-of-concept display prototype.
