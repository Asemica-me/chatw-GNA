import matplotlib.pyplot as plt
import numpy as np

def create_radar_diagram(
    categories,
    values,
    max_value=5,
    fill_color='blue',
    line_color='blue',
    alpha=0.25,
    line_thickness=0.5,
    label_distance=15,
    label_font_size=15
):

    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot the data
    ax.plot(angles, values, color=line_color, linewidth=line_thickness)
    ax.fill(angles, values, color=fill_color, alpha=alpha)

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=label_font_size)

    # Set the radial limits and add grid lines
    ax.set_ylim(0, max_value)
    ax.set_yticks(np.arange(1, max_value + 1)) # Set radial ticks at 1, 2, 3, 4, 5
    ax.set_yticklabels([str(i) for i in np.arange(1, max_value + 1)], color="grey", size=15)
    # Adjust the distance of the x and y axis labels
    ax.tick_params(axis='y', pad=1)
    ax.tick_params(axis='x', pad=label_distance)

    ax.grid(True)

    plt.show()

if __name__ == "__main__":

    categories = ['Consistency', 'Relevance', 'Fluency', 'Completeness']
    values = [5, 4.20, 4.80, 4.20]

    create_radar_diagram(
        categories,
        values,
        max_value=5,
        fill_color='blue',
        line_color='blue',
        alpha=0.25,
        line_thickness=0.5,
        label_distance=50,
        label_font_size=15
    )