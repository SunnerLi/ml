# HW1

[![Packagist](https://img.shields.io/badge/Python-3.5-blue.svg)]()
[![Packagist](https://img.shields.io/badge/Numpy-1.13.1-blue.svg)]()
[![Packagist](https://img.shields.io/badge/Scipy-0.19.1-blue.svg)]()
[![Packagist](https://img.shields.io/badge/Matplotlib-2.0.2-blue.svg)]()

Usage
---
Generate the random data point with specific target function
```
$ python generate.py --format sin --num 30   # Target function is cosine curve
$ python generate.py --format cos --num 30   # Target function is sine curve
$ python generate.py --format tan --num 30   # Target function is tangent curve
```

Print the result on the terminal toward given input data points file
```
python main.py --name ./data1.dat --lead 30 --lambda 0
```

See different result by different lambda factor
```
python main.py --name ./data1.dat --lead 30 
```

Issue
---
By the requirement, I should implement the inverse procedure by LU decomposition. However, I found there's a little difference after I adopt this method. Moreover, the result is a little shifted after I use LU decomposition. As the result, I still use LU decomposition to find the inverse in the single experiment. However, the numpy build-in inverse function will be used in the multiple experiments situation. 

<br/>

There's another matter. Even though I use `numpy` and `scipy` library, I just use it to do the fundamental operations (e.g. inverse, transpose). The `matplotlib` library is used to show the result. **The main process of linear regreesion isn't done by library**.     