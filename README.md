# iisignature-tensorflow-2

Simple tensorflow layer wrapper over the [iisignature library](https://github.com/bottler/iisignature/tree/master). 

The iisignature library examples for tensorflow covered only tensorflow1.X versions, this package only recreate the same examples for tensorflow 2.X versions.

A [pip package](https://pypi.org/project/iisignature_tensorflow_2/) can be installed directly using

```bash
pip install iisignature_tensorflow_2
```

The package implements SigLayer, LogSigLayer, SigScaleLayer and SigJoinLayer.

The SigJoinLayer not implements FixedGrad method for the moment.

For the first two they can be directly used in keras Sequential models, for the last two they need to be used in a functional model.

The package also includes a simple example of how you can use them (not specially smartly, just for the example).

For more details on how to use this refers to the original [iisignature library](https://github.com/bottler/iisignature/tree/master) or [paper](https://arxiv.org/abs/1802.08252)

It should be noted that the signature layer do not accept XLA compilation as the underlying operations are not made from such, any helps to implements this is welcome !

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
