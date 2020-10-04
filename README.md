# Fast Single Image Super-Resolution Using a New Analytical Solution for <a href="https://www.codecogs.com/eqnedit.php?latex=l_{2}&space;-&space;l_{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{2}&space;-&space;l_{2}" title="l_{2} - l_{2}" /></a>  Problems | Computer Vision

Centrale Lille | [Anas ESSOUNAINI](https://www.linkedin.com/in/anas-essounaini-b7514014a/) |  [Ali MOURTADA](https://www.linkedin.com/in/ali-mourtada-57714214a/) | [Imad AOUALI](https://www.linkedin.com/in/imad-aouali/)

***

## Description

This project aims to implement the algorithms described in the paper [Fast Single Image Super-Resolution Using a New Analytical Soltion for Problems](documents/paper_fast_super_resolution.pdf) in order to create __high resolution__ images from __low resolution__ ones. We have been able to reproduce the paper's results.

## Repository Structure 

This repository consists in several directories with specific purposes:

- `image_super_resolution`: This directory contains the `image_super_resolution` Python Library developed exclusively for this project. It contains functions used within the scripts. To install the library, navigates to the project directory and run the following `pip` command:
  ```Shell
  cd [path_to_project_directory]
  pip install -e .
  ```
  1. `utils`: module with intermediate functions
  2. `analytical`: implementation of FSR Analytical algorithm
  3. `iterative`: implementation of ADMM algorithm
- `prototype notebooks`: preliminary notebook for the project.
- `images`: images on which we apply high resolution algorithms.
- `test_package`: test of the algorithm on a picture (lena for instance)

---
For any information, feedback or questions, please [contact us][anas-email]

















[anas-email]: mailto:essounaini97@gmail.com
[ali-email]: mailto:mourtada.ali1997@gmail.com 