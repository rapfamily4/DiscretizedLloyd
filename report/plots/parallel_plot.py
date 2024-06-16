import plotly.graph_objects as go
import pandas as pd

target = 'time_avg'
df = pd.read_csv("./performance_test_16.csv", comment="#")

ln = dict(
    color = df[target],
    colorscale = [[0, '#47eabc'], [0.5, '#ffd700'], [1, '#ea4848']],
    showscale = True,
)
dim = list([
    dict(label = 'Graph Type',
         values = df['graph_type'],
         range = [0,4],
         tickvals = [0,1,2,3,4],
         ticktext = ["vertex","vertex dense","triangle","triangle dense","mixed"]),
    dict(label = "Greedy Mode",
         values = df['greedy_mode'],
         range = [0,2],
         tickvals = [0,1,2],
         ticktext = ["disabled","standard","extended"]),
    dict(label = "Optimize Macrostep",
         values = df['opt_macro'],
         range = [0,1],
         tickvals = [0,1],
         ticktext = ["off", "on"]),
    dict(label = "Optimize Microstep",
         values = df['opt_micro'],
         range = [0,1],
         tickvals = [0,1],
         ticktext = ["off", "on"]),
    dict(label = "Min Time (ms)", values = df['time_min'], visible = (target == 'time_min')),
    dict(label = "Average Time (ms)", values = df['time_avg'], visible = (target == 'time_avg')),
    dict(label = "Max Time (ms)", values = df['time_max'], visible = (target == 'time_max')),
    dict(label = "Min Partition Iterations", values = df['part_min'], visible = (target == 'part_min')),
    dict(label = "Average Partition Iterations", values = df['part_avg'], visible = (target == 'part_avg')),
    dict(label = "Max Partition Iterations", values = df['part_max'], visible = (target == 'part_max')),
    dict(label = "Min Greedy Steps", values = df['greedy_min'], visible = (target == 'greedy_min')),
    dict(label = "Average Greedy Steps", values = df['greedy_avg'], visible = (target == 'greedy_avg')),
    dict(label = "Max Greedy Steps", values = df['greedy_max'], visible = (target == 'greedy_max')),
    dict(label = "Min Precise Macroteps", values = df['precise_min'], visible = (target == 'precise_min')),
    dict(label = "Average Precise Macroteps", values = df['precise_avg'], visible = (target == 'precise_avg')),
    dict(label = "Max Precise Macroteps", values = df['precise_max'], visible = (target == 'precise_max'))
])

fig = go.Figure(data = go.Parcoords(dimensions = dim, line = ln))
fig.show()