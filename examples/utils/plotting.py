"""
Nixtla Brand Theme for Matplotlib Plots

Provides consistent styling for Nixtla/TimeGPT related visualizations.
"""


def apply_nixtla_style(ax, fig=None, title=None, xlabel=None, ylabel=None, legend=True):
    """
    Apply Nixtla brand styling to matplotlib plots

    Parameters:
    - ax: matplotlib axes object
    - fig: matplotlib figure object (optional, will be inferred if not provided)
    - title: plot title (optional)
    - xlabel: x-axis label (optional)
    - ylabel: y-axis label (optional)
    - legend: whether to style the legend (default: True)
    """
    # Nixtla brand colors
    NIXTLA_COLORS = {
        "background": "#160741",
        "text": "#FFFFFF",
        "primary": "#98FE09",  # lime
        "secondary": "#02FEFA",  # cyan
        "accent": "#0E00F8",  # bright blue
    }

    # Get figure if not provided
    if fig is None:
        fig = ax.get_figure()

    # Apply background colors
    ax.set_facecolor(NIXTLA_COLORS["background"])
    fig.patch.set_facecolor(NIXTLA_COLORS["background"])

    # Style text elements
    ax.tick_params(colors=NIXTLA_COLORS["text"])

    # Apply labels if provided
    if title:
        ax.set_title(title, color=NIXTLA_COLORS["text"], fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=NIXTLA_COLORS["text"])
    if ylabel:
        ax.set_ylabel(ylabel, color=NIXTLA_COLORS["text"])

    # Style legend if present and requested
    if legend and ax.get_legend():
        legend_obj = ax.get_legend()
        legend_frame = legend_obj.get_frame()
        legend_frame.set_facecolor(NIXTLA_COLORS["background"])
        legend_frame.set_edgecolor(NIXTLA_COLORS["text"])
        for text in legend_obj.get_texts():
            text.set_color(NIXTLA_COLORS["text"])

    # Style spines
    for spine in ax.spines.values():
        spine.set_color(NIXTLA_COLORS["text"])
        spine.set_linewidth(0.5)

    return ax


def get_nixtla_colors():
    """
    Get Nixtla brand colors dictionary

    Returns:
    - dict: Color palette with keys: background, text, primary, secondary, accent
    """
    return {
        "background": "#160741",  # dark purple
        "text": "#FFFFFF",  # white
        "primary": "#98FE09",  # lime (main brand color)
        "secondary": "#02FEFA",  # cyan (secondary accent)
        "accent": "#0E00F8",  # bright blue (additional accent)
    }
