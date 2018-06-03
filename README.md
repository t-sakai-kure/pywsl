# pywsl: **py**thon codes for **w**eakly-**s**upervised **l**earning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/t-sakai-kure/pywsl.svg?branch=master)](https://travis-ci.org/t-sakai-kure/pywsl)
[![PyPI version](https://badge.fury.io/py/pywsl.svg)](https://badge.fury.io/py/pywsl)

This package contains the following implementation:
- ***Unbiased PU learning***  
  &nbsp;&nbsp;&nbsp; in "Convex formulation for learning from positive and unlabeled data", ICML, 2015 [[uPU]](#uPU)
- ***Non-negative PU Learning***  
  &nbsp;&nbsp;&nbsp; in "Positive-unlabeled learning with non-negative risk estimator", NIPS, 2017 [[nnPU]](#nnPU)
- ***PU Set Kernel Classifier***  
  &nbsp;&nbsp;&nbsp; in "Convex formulation of multiple instance learning from positive and unlabeled bags", Neural Networks, 2018 [[PU-SKC]](#pu-skc)
- ***Class-prior estimation based on energy distance***  
  &nbsp;&nbsp;&nbsp; in "Computationally efficient class-prior estimation under class balance change using energy distance", IEICE-ED, 2016 [[CPE-ENE]](#cpe_ene).
- ***PNU classification***  
  &nbsp;&nbsp;&nbsp; in "Semi-supervised classification based on classification from positive and unlabeled data", ICML 2017 [[PNU]](#pnu_mr).
- ***PNU-AUC optimization***  
  &nbsp;&nbsp;&nbsp; in "Semi-supervised AUC optimization based on positive-unlabeled learning", MLJ 2018 [[PNU-AUC]](#pnu_auc).

## Installation
```sh
$ pip install pywsl
```

## Contributors
- [Tomoya Sakai](https://t-sakai-kure.github.io)
- [Han Bao](http://levelfour.github.io)
- [Ryuichi Kiryo](https://github.com/kiryor)

## References
1. <a name="uPU"> du Plessis, M. C., Niu, G., and Sugiyama, M.   
  Convex formulation for learning from positive and unlabeled data.   
  In Bach, F. and Blei, D. (Eds.), Proceedings of 32nd International Conference on Machine Learning,
  JMLR Workshop and Conference Proceedings, vol.37, pp.1386-1394, Lille, France, Jul. 6-11, 2015. 
1. <a name="nnPU"> Kiryo, R., Niu, G., du Plessis, M. C., and Sugiyama, M.   
  Positive-unlabeled learning with non-negative risk estimator.  
  In Guyon, I.,  Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (Eds.), 
  Advances in Neural Information Processing Systems 30, pp.1674-1684, 2017.   
1. <a name="pu-skc"> Bao, H., Sakai, T., Sato, I., and Sugiyama, M.  
  Convex formulation of multiple instance learning from positive and unlabeled bags.  
  Neural Networks, vol.105, pp.132-141, 2018.  
1. <a name="cpe_ene"> Kawakubo, H., du Plessis, M. C., and Sugiyama, M.  
  Computationally efficient class-prior estimation under class balance change using energy distance.   
  IEICE Transactions on Information and Systems, vol.E99-D, no.1, pp.176-186, 2016.
1. <a name="pnu_mr"> Sakai, T., du Plessis, M. C., Niu, G., and Sugiyama, M.   
  Semi-supervised classification based on classification from positive and unlabeled data.   
  In Precup, D. and Teh, Y. W. (Eds.), Proceedings of 34th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol.70, pp.2998-3006, Sydney, Australia, Aug. 6-12, 2017.  
1. <a name="pnu_auc"> Sakai, T., Niu, G., and Sugiyama, M.   
  Semi-supervised AUC optimization based on positive-unlabeled learning.   
  Machine Learning, vol.107, no.4, pp.767-794, 2018.   
