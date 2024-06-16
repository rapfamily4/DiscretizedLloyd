import plotly.express as px
import pandas as pd

use_minutes = False
df = pd.read_csv("./runs_log.csv", comment="#")

lbl ={
    'seeds':'Seed Count',
    'run_time':'Running Time (minutes)' if use_minutes else 'Running Time (ms)',
    'greedy_steps':'Greedy Steps',
    'avg_greedy_time':'Average Greedy Step Time (minutes)' if use_minutes else 'Average Greedy Step Time (ms)',
    'precise_macrosteps':'Precise Macrosteps',
    'avg_precise_time':'Average Precise Macrostep Time (minutes)' if use_minutes else 'Average Precise Macrostep Time (ms)',
    'partition_iters':'Partition Iterations',
    'avg_partition_time':'Average Partitioning Time (minutes)' if use_minutes else 'Average Partitioning Time (ms)'
}

if use_minutes:
    df['run_time'] /= 60000
    df['avg_greedy_time'] /= 60000
    df['avg_precise_time'] /= 60000
    df['avg_partition_time'] /= 60000

fig = px.bar(df, x='seeds', y='run_time', labels=lbl)
#fig = px.bar(df, x='seeds', y='partition_iters', labels=lbl)
#fig = px.bar(df, x='seeds', y='avg_partition_time', labels=lbl)
#fig = px.bar(df, x='seeds', y=['greedy_steps','precise_macrosteps'], labels=lbl)
#fig = px.bar(df, x='seeds', y=['avg_greedy_time','avg_precise_time'], labels=lbl)
fig.update_xaxes(type='category')
fig.show()