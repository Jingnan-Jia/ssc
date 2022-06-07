'''
Bland-Altman mean-difference plots

Author: Joses Ho
License: BSD-3
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def create_mpl_ax(ax=None):
    """Helper function for when a single plot axis is needed.

    Parameters
    ----------
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    ax : AxesSubplot
        The created axis if `ax` is None, otherwise the axis that was passed
        in.

    Notes
    -----
    This function imports `matplotlib.pyplot`, which should only be done to
    create (a) figure(s) with ``plt.figure``.  All other functionality exposed
    by the pyplot module can and should be imported directly from its
    Matplotlib module.

    See Also
    --------
    create_mpl_fig

    Examples
    --------
    A plotting function has a keyword ``ax=None``.  Then calls:

    >>> from statsmodels.graphics import utils
    >>> fig, ax = create_mpl_ax(ax)
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    return fig, ax


def mean_diff_plot(m1, m2,
                   sd_limit=1.96,
                   ax=None,
                   scatter_kwds=None,
                   mean_line_kwds=None,
                   limit_lines_kwds=None,
                   x_is_label=True,
                   bland_in_1_mean_std=None,
                   adap_markersize=1,
                   ynotdiff=False):
    """
    Construct a Tukey/Bland-Altman Mean Difference Plot.

    Tukey's Mean Difference Plot (also known as a Bland-Altman plot) is a
    graphical method to analyze the differences between two methods of
    measurement. The mean of the measures is plotted against their difference.

    For more information see
    https://en.wikipedia.org/wiki/Bland-Altman_plot

    Parameters
    ----------
    m1 : array_like
        A 1-d array.
    m2 : array_like
        A 1-d array.
    sd_limit : float
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted are md +/- sd_limit * sd.
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences. If sd_limit = 0, no limits will be plotted, and
        the ylimit of the plot defaults to 3 standard deviations on either
        side of the mean.
    ax : AxesSubplot
        If `ax` is None, then a figure is created. If an axis instance is
        given, the mean difference plot is drawn on the axis.
    scatter_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method
    mean_line_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
    limit_lines_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    References
    ----------
    Bland JM, Altman DG (1986). "Statistical methods for assessing agreement
    between two methods of clinical measurement"

    Examples
    --------

    Load relevant libraries.

    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Making a mean difference plot.

    >>> # Seed the random number generator.
    >>> # This ensures that the results below are reproducible.
    >>> np.random.seed(9999)
    >>> m1 = np.random.random(20)
    >>> m2 = np.random.random(20)
    >>> f, ax = plt.subplots(1, figsize = (8,5))
    >>> sm.graphics.mean_diff_plot(m1, m2, ax = ax)
    >>> plt.show()

    """
    # fontsize = 20
    fig, ax = create_mpl_ax(ax)

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    if adap_markersize:
        if ynotdiff:
            s_candidate = m1.reshape(-1, )
        else:
            s_candidate = diffs.reshape(-1, )
        values, counts = np.unique(s_candidate, return_counts=True)
        counts_dt = dict(zip(values, counts))
        s_linear = np.array([abs(counts_dt[i]) *4 for i in s_candidate])
    else:
        s_linear = None
    if scatter_kwds is None:
        scatter_kwds = {"s": s_linear}
    else:
        scatter_kwds.update({"s": s_linear})
    # scatter_kwds = scatter_kwds or {"s": s_linear}
    if 's' not in scatter_kwds or scatter_kwds['s'] is None:
        scatter_kwds['s'] = 4
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'black'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    if ynotdiff:
        if x_is_label:
            ax.scatter(m2.reshape(-1, ), m1, **scatter_kwds)

        else:
            ax.scatter(means.reshape(-1, ), m1, **scatter_kwds)
    else:
        if x_is_label:
            ax.scatter(m2.reshape(-1, ), diffs, **scatter_kwds)
        else:
            ax.scatter(means.reshape(-1, ), diffs, **scatter_kwds)

    if bland_in_1_mean_std is not None:
        mean_diff = bland_in_1_mean_std['mean']
        std_diff = bland_in_1_mean_std['std']
    if sd_limit > 0:
        ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)

        # ax.axhline(md + limitOfAgreement * sd, color=loaColour, linestyle='--')
        # ax.axhline(md - limitOfAgreement * sd, color=loaColour, linestyle='--')
        # ax.axhline(lim, **limit_lines_kwds)

        trans = transforms.blended_transform_factory(
            ax.transAxes, ax.transData)

        limitOfAgreementRange = (mean_diff + (sd_limit * std_diff)) - (mean_diff - sd_limit * std_diff)
        offset = (limitOfAgreementRange / 100.0) * 1.5

        ax.text(0.98, mean_diff + offset, 'Mean', ha="right", va="bottom", transform=trans, fontsize='large')
        ax.text(0.98, mean_diff - (2*offset), f'{mean_diff:.2f}', ha="right", va="top", transform=trans, fontsize='large')

        ax.text(0.98, mean_diff + (sd_limit * std_diff) + offset, f'+{sd_limit:.2f} SD', ha="right", va="bottom",
                transform=trans, fontsize='large')
        ax.text(0.98, mean_diff + (sd_limit * std_diff) - (2*offset), f'{mean_diff + sd_limit * std_diff:.2f}', ha="right", va="top",
                transform=trans, fontsize='large')

        ax.text(0.98, mean_diff - (sd_limit * std_diff) - (2* offset), f'-{sd_limit:.2f} SD', ha="right", va="top",
                transform=trans, fontsize='large')
        ax.text(0.98, mean_diff - (sd_limit * std_diff) + offset, f'{mean_diff - sd_limit * std_diff:.2f}', ha="right", va="bottom",
                transform=trans, fontsize='large')
        # ax.text(0.05, 0.9 * (2*half_ylim), 'A)', ha="left", fontsize='xx-large', transform=trans)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    # if ynotdiff:
    #     ax.set_ylabel('$Obs2_{T2} \;score\;(\%)$', fontsize=15)
    #     ax.set_xlabel('GT score (%)', fontsize=15)
    # else:
    #     ax.set_ylabel('$Obs2_{T2}\;-\;GT\;score\;(\%)$', fontsize=15)
    #     ax.set_xlabel('Average score (%)', fontsize=15)

    if ynotdiff:
        ax.set_ylabel('L-Net (slice number)', fontsize=15)
        ax.set_xlabel('Ground truth (slice number)', fontsize=15)
    else:
        ax.set_ylabel('L-Net - Ground truth (slices)', fontsize=15)
        ax.set_xlabel('Average (slice number)', fontsize=15)

    ax.tick_params(labelsize=13)
    fig.tight_layout()
    return fig
