import matplotlib.pyplot as plt
import json

# Example data
for q in range(4):
  with open(f"graphout{q}.json", "r") as file:
    data = json.load(file)
  """data = {
      "Series1": [[10, 20, 30, 40, 50]],
      "Series2": [[15, 25, 35, 45, 55]],
      "Series3": [[20, 30, 40, 50, 60]],
      "Baseline": [[5, 15, 25, 35, 45, 46]]
  }"""

  # Create a figure and axis
  fig, ax = plt.subplots()

  ally = []

  # Plot each series
  for i, (series_name, series_data) in enumerate(data.items()):
      label = series_name
      
      x = list(range(1, len(series_data[0]) + 1))  # x-values
      y = [val for val in series_data[0]]  # y-values multiplied by 100
      print(max(y))
      ally += y
      ax.plot(x, y, label=label)

  # Add legend
  ax.legend()
  ax.legend(loc='lower right')

  ax.set_yticks(range((int(min(ally))//5)*5, int(max(ally)) + 1, 5))
  ax.set_xticks(range(0, 120, 10))

  ax.grid(color='gray', linestyle=':', linewidth=0.5)

  # Set labels and title
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Validation accuracy / %")
  ax.set_title(f"Validation accuracy against epochs at {(q+1)*20}% noise")

  # Show the plot
  plt.savefig(f'{(q+1)*20}%.png')
  #plt.show()