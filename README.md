# pad
PAD is a novel privacy-preserving data publication system that leverages interactions with data users to improve utility of privatized datasets.

The repository contains the source code and demos of PAD in Python.

For a detailed description of the algorithms, see the paper (https://ruoxijia.github.io/assets/papers/pad-buildsys-final.pdf).
To cite the paper:

```
@inproceedings{pad2017,
  author = {Jia, Ruoxi and Caleb Sangogboye, Fisayo and Hong, Tianzhen and Spanos, Costas and Baun Kj{\ae}rgaard, Mikkel},
  title = {PAD: Protecting Anonymity in Publishing Building Related Datasets},
  booktitle={Proceedings of the 4th ACM Conference on Embedded Systems for Energy-Efficient Buildings},
  year={2017},
  organization={ACM}
}
```

## Extension for the adoption of the PAD framework to traffic data
For the application of the PAD framework the file [exp_opa_tad.py](experiment/opa_tad/exp_opa_tad.py) was created. Arguments can be added to this file. The list of parameters is described [here](#console-arguments). The help can be displayed with the ` -h` argument.

### Console Arguments
* `-h`: Displays the help on the console
* `-i <input file>`: Specification of the input file for the experiment      
* `-m <mc num>`: Specification of the number of Monte Carlo simulations
* `-c <number of classes>`: Specifies the number of classes in which the data is to be grouped
* `-p`: If this argument is used, the data is pre-sanitized
* `-v`: If this argument is specified the class is used for the number of vehicles
