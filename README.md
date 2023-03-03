# sweep-design

Simple way to create sweep signal.

## The project is intended for designing sweep signals.

The package is intended to create and develop sweep signals of
varying complexity.

The project can be used both for educational and work purposes.

It is convenient to use [`Jupyter Lab`](https://jupyter.org/) or
[`Jupyter Notebook`](https://jupyter.org/) to speed up the development
of signals, to compare their parameters with other signals,
and to visualize them.

The project is designed so that you can easily change the creation of sweep
signals. For example, write your own methods describing how the frequency
and amplitude will change from the time of the sweep signal.

The project was made to be able to create various sweep signals: implemented
and not implemented by a vibration source, from simple ones, like a linear
sweep signal, to complex ones, like a pseudo-random sweep signal.

Tools have been written with which unrealizable sweep signals
could be made realizable.

In addition, documentation consist tutorial how to work with library
and examples of ready-made sweeps. You can write own sweep creation.

# Installation

To install use:

```bash
$ pip install sweep-design
```

or using `poetry`

```bash
$ poetry add sweep-design
```

Also you can clone or load project from [GitHub](https://github.com/Omnivanitate/sweep-design),
and install requirement packages using the

```bash
$ pip install -r requirement.txt
```

or if you want develop, use

```bash
$ pip install -r requirement-dev.txt
```

or

```bash
$ poetry install
```

or coping pieces of code and create your own.

## Usage

The project is a library. Working with it is the same as with
other third-part libraries of the python language.  
An example of how to include the library is described
[here](https://docs.python.org/3/tutorial/modules.html).

The library consists sub-modules:

- `sweep_design.config` - contains the project configuration `Config` and `SweepConfig`.
- `sweep_design.defaults` - contains default methods to calculate.
- `sweep_design.prepared_sweep` - contains sweep signal templates.
- `sweep_design.utility_functions` - contains function to work with signals.
- `sweep_design.core` - contains basic classes `MathOperation` and `RelationProtocol`.
- `sweep_design.exc` - contains exceptions.
- `sweep_design.axis` - contains class `ArrayAxis`
- `sweep_design.relation ` - contains class `Relation`
- `sweep_design.signal ` - contains class `Signal`
- `sweep_design.spectrum ` - contains class `Spectrum`
- `sweep_design.sweep` - contains class `Sweep`
- `sweep_design.uncalculated` - contains classes `UncalculatedSweep` and `ApriorUncalculatedSweep`
- `sweep_design.spectrogram` - contains classes `Spectrogram`

For convenient base classes:
`ArrayAxis`, `Relation`, `Signal`, `Spectrum`, `Sweep`, `UncalculatedSweep`,
`ApriorUncalculatedSweep`, `Config`, `ConfigSweep` - can be imported from
a `sweep_design` module.

For example:

```python
from sweep_design import Signal
```

Utility functions can be imported from `sweep_design.utility_functions`.  
And prepared sweep - from `sweep_design.prepared_sweep`.

### Quick start. Simple work flow.

Below is a simple example of creating a sweep signal and visualizing it.
A more extended description of the work of the library in the documentation.
Other examples are contained in the examples contains in _Tutorial_ and
_Prepared sweep_ sections.

For the following code [`Matplotlib`](https://matplotlib.org/) need be used
to visualize a result of work. But `Matplotlib` can be replaced with another
library that you use.

```python
import matplotlib.pyplot as plt

from sweep_design import ArrayAxis, UncalculatedSweep

time = ArrayAxis(start=0., end=10., sample=0.01)

usw = UncalculatedSweep(time=time)
sw = usw()

t_sw, a_sw = sw.get_data()
plt.plot(t_sw, a_sw)
plt.xlabel('Time, s')
plt.ylabel('Amplitude')
plt.title('Sweep-signal')
```

Result:

![sweep_with_matplotlib](https://user-images.githubusercontent.com/89973180/156033978-ccc8de40-9f6b-4bb1-b59f-7a3ea41d2f64.png "Linear Sweep")

## Credits

`sweep-design` was created with  
[`numpy`](https://numpy.org/)  
[`scipy`](https://scipy.org/)  
[`EMD-signal`](https://pyemd.readthedocs.io/en/latest/)

### TODO

1. Merge array axis.
