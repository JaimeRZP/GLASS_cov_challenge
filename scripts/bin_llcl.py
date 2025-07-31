import numpy as np


def bin_llcl(llcl_in, ubin, flatten=False, uniform=False):
    """x, y, dx, dy = bin_llcl( llcl_in, bin, flatten=False, uniform=False)

    turns llcl_in (= continuous l*(l+1)*cl/2Pi) into a binned version with
       a constant or variable binwidth 'bin'


    INPUTS
      llcl_in : input l*(l+1)*Cl/2Pi, 1D vector, defined for each l from l=0
      bin : *integer*, can be either a scalar = dl
        or a vector defining the bins boundaries :
        [low0, low1, low2, ...,low(n-2), low(n-1)+1]

    OUTPUTS
        x:  center of bins
        y:  binned l*(l+1)*Cl/2Pi
        dx: width of each bin
        dy: returns on output the rms of C(l) for a full sky observation
           = C(l) * sqrt( 2/ 2l+1 / dl)


    KEYWORDS
      flatten: if set, the input C(l) is multiplied internally by l*(l+1)/2Pi
        before being binned. By default, the input C(l) is binned as is.

      uniform: if set, each l is given the same weight in the bin.
        By default, a weighting propto (2*l+1) (inverse cosmic
        variance) is applied to each l.
        In any cases, the output x is the same

    HISTORY
      2016-11-23: adapted from Healpix/IDL    bin_llcl.pro
      2021-05-26: enforce integer bins, ready for python3; ran by "black"
      2024-04-26: (re)fixed crash-causing bug in irregular binning case
    """

    #bin = np.copy(ubin)
    bin = np.int32(ubin)
    nb = np.size(bin)
    lmax_in = np.size(llcl_in) - 1
    if nb > 1:
        k = np.where(bin <= (lmax_in + 1))[0]
        nk = np.size(k)
        if nk == (nb - 1):
            bin = np.minimum(bin, lmax_in + 1)  # shorten last bin
        if nk < (nb - 1):
            # shorten last valid bins, and drop the ones beyond lmax
            bin = np.concatenate((bin[k], [lmax_in + 1]))
            nb = nk + 1

    if nb == 1:
        # regular binning
        nbins = np.int32(lmax_in / bin)
        lmax = nbins * bin - 1
        l = np.arange(lmax + 1, dtype=np.float64)
        w = 2 * l + 1
        if uniform:
            w = np.ones(lmax + 1, dtype=np.float64)
        y = np.copy(llcl_in[0 : lmax + 1])
        if flatten:
            y *= l * (l + 1.0) / (2 * np.pi)
        w1 = np.reshape(w, (nbins, bin))
        y1 = np.reshape(y * w, (nbins, bin))
        l1 = np.reshape(l, (nbins, bin))
        n1 = np.ones((nbins, bin))

        llcl_out = np.sum(y1, 1) / np.sum(w1, 1)
        l_out = np.sum(l1, 1) / np.sum(n1, 1)
        dl = bin * np.ones(nbins, dtype=np.int32)

    else:

        # irregular binning
        lmax = np.int32(np.max(bin)) - 1
        nbins = nb - 1
        good = np.where(bin < lmax)[0]
        ng = np.size(good)
        if ng == 0:
            print("l-range of binning does not intersect that of data")
            return -1, -1, -1, -1

        l = np.arange(lmax + 1, dtype=np.float64)
        w = 2 * l + 1
        if uniform:
            w = np.ones(lmax + 1, dtype=np.float64)
        y = np.copy(llcl_in[0 : lmax + 1])
        if flatten:
            y *= l * (l + 1.0) / (2 * np.pi)
        l_out = np.zeros(nbins, dtype=np.float64)
        llcl_out = np.zeros(nbins, dtype=np.float64)
        dl = np.zeros(nbins, dtype=np.int32)
        for i in range(nbins):
            l_out[i] = np.mean(l[bin[i] : bin[i + 1]])
            dl[i] = bin[i + 1] - bin[i]
            llcl_out[i] = np.sum((y * w)[bin[i] : bin[i + 1]]) \
                        / np.sum(  w    [bin[i] : bin[i + 1]])

    dllcl = llcl_out * np.sqrt(2 / (2 * l_out + 1.0) / dl)

    return l_out, llcl_out, dl, dllcl
