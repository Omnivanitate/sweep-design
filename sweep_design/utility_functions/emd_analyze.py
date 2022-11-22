from typing import Any, List

from PyEMD import CEEMDAN, EMD  # type: ignore
from ..signal import Signal


def get_IMFs_ceemdan(
    data: Signal,
    number_seedman=30,
    epsilon=0.005,
    ext_EMD: Any = None,
    parallel=False,
    processes=None,
    noise_scale=1,
    noise_kind="normal",
    range_thr=0.01,
    total_power_thr=0.05,
) -> List[Signal]:
    '''Empirical mode decomposition (EMD).
    Using CEEMDAN from PyEMD (https://pyemd.readthedocs.io/) to calculate IMFs
    and return them as a list of `Signals`.

    About empirical mode decomposition on
    https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform#Techniques

    Args:
        data (Signal): _description_

        number_seedman (int, optional): _description_. Defaults to 30.

        epsilon (float, optional): Scale for added noise (\\epsilon) which
        multiply std \\sigma: \beta = \\epsilon \\cdot \\sigma.
        Defaults to 0.005.

        ext_EMD (Any, optional): One can pass EMD object defined outside,
        which will be used to compute IMF decompositions in each trial.
        If none is passed then EMD with default options is used.
        Defaults to None.

        parallel (bool, optional): Flag whether to use multiprocessing in
        EEMD execution. Since each EMD(s+noise) is independent this should
        improve execution speed considerably. *Note* that it's disabled by
        default because it's the most common problem when CEEMDAN takes too
        long time to finish. If you set the flag to True, make also sure to
        set processes to some reasonable value. Defaults to False.

        processes (_type_, optional): Number of processes harness when
        executing in parallel mode. The value should be between 1 and
        max that depends on your hardware. Defaults to None.

        noise_scale (int, optional): Scale (amplitude) of the added noise.
        Defaults to 1.

        noise_kind (str, optional):  What type of noise to add. Allowed
        are "normal" (default) and "uniform". Defaults to "normal".

        range_thr (float, optional):  Range threshold used as an IMF check.
        The value is in percentage compared to initial signal's amplitude.
        If absolute amplitude (max - min) is below the range_thr then the
        decomposition is finished. Defaults to 0.01.

        total_power_thr (float, optional): Signal's power threshold.
        Finishes decomposition if sum(abs(r)) < thr. Defaults to 0.05.



    Returns:
        List[Signal]: List of Signals expected IMFs.
    '''
    y = data.y
    emd = CEEMDAN(
        number_seedman,
        epsilon,
        ext_EMD,
        parallel,
        processes=processes,
        noise_scale=noise_scale,
        noise_kind=noise_kind,
        range_thr=range_thr,
        total_power_thr=total_power_thr,
    )
    IMFs = emd(y)
    result = [Signal(data.x.copy(), k) for k in IMFs]
    return result


def get_IMFs_emd(data: Signal) -> List[Signal]:
    '''"Empirical mode decomposition (EMD).

    Using EMD from PyEMD (https://pyemd.readthedocs.io/) to calculate IMFs
    and return them as a list of `Signals`.

    About empirical mode decomposition on
    https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform#Techniques

    Args:
        data (Signal): signal to calculate IMFs.

    Returns:
        List[Signal]: List of Signals expected IMFs.
    '''
    y = data.y
    emd = EMD()
    IMFs = emd(y)
    result = [Signal(data.x.copy(), k) for k in IMFs]
    return result
