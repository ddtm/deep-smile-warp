# DeepWarp for Facial Expression Manipulation

**Warning: the repo is under construction so the code may not be runnable out of the box!**

This is a repository containing `torch7` code implementing
[DeepWarp](http://sites.skoltech.ru/compvision/projects/deepwarp/files/deepwarp_eccv2016.pdf). Be sure to check out the
[online demo](http://163.172.78.19/). Due to licensing restrictions
I'm not able to release the original gaze manipulation code and corresponding dataset, but the model presented here
is very similar and you are free to use it as a reference point.

## Showcase

To showcase general applicability of the approach, I'm using an ever so slightly modified network to handle smile addition and removal
in the images containg human faces.

## Citation

Please cite **DeepWarp** in your publications if it helps your research:

    @inproceedings{ganin2016deepwarp,
      title={DeepWarp: Photorealistic image resynthesis for gaze manipulation},
      author={Ganin, Yaroslav and Kononenko, Daniil and Sungatullina, Diana and Lempitsky, Victor},
      booktitle={European Conference on Computer Vision},
      pages={311--326},
      year={2016},
      organization={Springer}
    }
