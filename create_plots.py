import argparse
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

PLOT_WIDTH = 900
PLOT_HEIGHT = PLOT_WIDTH // 3

LAM_COL = [
    '#009ee3', #0 aqua
    '#983082', #1 fresh violet
    '#ffbc29', #2 sunshine
    '#35cdb4', #3 carribean
    '#e82e82', #4 fuchsia
    '#59bdf7', #5 sky blue
    '#ec6469', #6 indian red
    '#706f6f', #7 gray
    '#4a4ad8', #8 corn flower
    '#0c122b', #9 dark corn flower
    '#ffffff'
]
CC_COL = ['#c3f436', '#044454']

SEL_COLORS = [CC_COL[1], LAM_COL[3], LAM_COL[8], LAM_COL[1], LAM_COL[6], LAM_COL[2]]
pio.templates[pio.templates.default].layout.colorway = SEL_COLORS

def print_init(fname):
    print(f'                 - -- ---  {fname:<20}  --- -- -                 ')
    return fname

def finalize(fig, fname, show, x_scale=1, y_scale=1, tshift=0, yshift=0):
    fig.update_layout(font_family='Open-Sherif', margin={'l': 0, 'r': 0, 'b': 0, 't': tshift},
                      width=PLOT_WIDTH*x_scale, height=PLOT_HEIGHT*y_scale)
    fig.update_annotations(yshift=2+yshift) # to adapt tex titles
    if show:
        fig.show()
    fig.write_image(f"{fname}.pdf")

def parse_param_count(s):
    if isinstance(s, str):
        s = s.strip()
        if s.endswith('B'):
            return float(s[:-1]) * 1e9
        elif s.endswith('M'):
            return float(s[:-1]) * 1e6
        elif s.endswith('K'):
            return float(s[:-1]) * 1e3
        else:
            try:
                return float(s)
            except Exception:
                return s
    return s

if __name__ == "__main__":
    results = []
    for fname in os.listdir(os.path.join(os.path.dirname(__file__), 'results')):
        if 'image_analysis.csv' in fname:
            # load results
            video_results = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results', fname))
            db_orig = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results', fname.replace('_image_analysis.csv', '.csv')))
            db_orig = db_orig.dropna().set_index('run_id').sort_values('start_time')
            properties = [col.replace('metrics.', '') for col in db_orig.columns if 'metrics' in col]
            db = db_orig.drop(columns=[col for col in db_orig.columns if 'params' not in col and 'metrics' not in col]) # drop everything that was not logged
            db = db.rename(columns=lambda col: col.replace('params.', '').replace('metrics.', ''))
            exp_name = fname.split('_')[0]
            db['experiment'] = exp_name
            assert video_results.shape[0] == db.shape[0] * 2 # should have twice as many entries
            # unify results for comparisons
            if exp_name == 'ollama': # replace 20.9B 8.2B 1B 3B etc with numbers
                db['parameters'] = db['parameters'].apply(parse_param_count)
            db['externally_measured_total'] = video_results.iloc[1::2]['val_diff'].values * 3.6e6
            results.append(db)
    # merge results
    db = pd.concat(results, ignore_index=True).rename(mapper=lambda col: col.replace('power_draw', 'codecarbon'), axis=1)
    db['static_estimate_total'] = db['running_time_total'] * 300 # based on https://mlco2.github.io/impact/ info for RTX 4090
    for col in ['static_estimate_total', 'externally_measured_total']:
        db[col.replace('_total', '')] = db[col] / db['n_samples']
    # calculate consumption per minute and differences
    for field in ['static_estimate', 'codecarbon', 'externally_measured']:
        db[f'{field}_per_min'] = db[f'{field}_total'] / db['running_time_total']
    for field in ['static_estimate', 'codecarbon']:
        db[f'{field}_diff'] = db['externally_measured'] - db[field]
        db[f'{field}_total_diff'] = db['externally_measured_total'] - db[f'{field}_total']
        db[f'{field}_rel_diff'] = db[f'{field}_diff'] / db['externally_measured'] * 100
        db[f'{field}_total_rel_diff'] = db[f'{field}_total_diff'] / db['externally_measured_total'] * 100
        assert np.all(np.isclose(db[f'{field}_rel_diff'], db[f'{field}_total_rel_diff']))
    # group results
    db_gpu = db[db['architecture'].str.contains('NVIDIA')]
    grouped_gpu = db_gpu.sort_values('externally_measured').groupby('model').first().sort_values('parameters')
    db_gpu_ollama = grouped_gpu[grouped_gpu['experiment'] == 'ollama']
    db_gpu_imagenet = grouped_gpu[grouped_gpu['experiment'] == 'imagenet']

    # init plotting
    os.makedirs('figures', exist_ok=True)
    os.chdir('figures')
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    os.remove("dummy.pdf")

    # opener plot
    fname = print_init('opener')
    traces = []
    for text, col, c in [['Static (ML Impact Calculator)', 'static_estimate', SEL_COLORS[1]], ['Dynamic (CodeCarbon)', 'codecarbon', SEL_COLORS[0]]]:
        for name, t_db, s in [['Vision', db_gpu_imagenet, 'x'], ['Language', db_gpu_ollama, 'circle']]:
            traces.append(go.Scatter(
                x=t_db['parameters'], y=t_db[f'{col}_rel_diff']*-1, mode='markers', marker={'symbol': s, 'color': c},
                name=name, legendgroup=text, legendgrouptitle={'text': text}
            ))
    fig = go.Figure(traces)
    fig.add_hline(y=0, line_dash="dot", annotation_text="Ground-Truth Energy Consumption")
    fig.update_yaxes(title='Over- / Underestimation [%]')
    fig.update_xaxes(title='Number of Model Parameters', type="log")
    fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="center", x=0.5, orientation='h'))
    finalize(fig, fname, show=True, x_scale=0.5)

    # groundtruth power consumption
    fname = print_init('groundtruth_power')
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for name, t_db, s, c in [['Vision', db_gpu_imagenet, 'x', SEL_COLORS[0]], ['Language', db_gpu_ollama, 'circle', SEL_COLORS[1]]]:
        for row, field in enumerate(['externally_measured_per_min', 'externally_measured']):
            fig.add_trace(go.Scatter(x=t_db.index, y=t_db[field], name=name, mode='markers', marker={'symbol': s, 'color': c}, showlegend=row==0),
                          row=1+row, col=1)
    # Set x-axis to categorical for the second row
    fig.update_xaxes(type='category', range=[-0.8, grouped_gpu.shape[0]-0.2], row=2, col=1)
    fig.update_yaxes(title='Energy Draw per Minute [W]', type="log", row=1, col=1)
    fig.update_yaxes(title='Energy Draw per Sample [W]', type="log", row=2, col=1)
    fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="left", x=0, orientation='h'))
    finalize(fig, fname, show=True, y_scale=2)

    grouped = db.groupby(['nogpu', 'model']).first().sort_values('parameters')
    texts = grouped.loc[0].index # .map(lambda v: '' if v in ['DenseNet121', 'EfficientNetB1', 'EfficientNetV2B0', 'EfficientNetV2B3', 'ResNet50V2', 'ResNet152V2'] else v)

    # power draw per model for all setups
    fig = go.Figure([
        go.Scatter(x=grouped.loc[0].index, y=grouped.loc[0,'power_draw'], mode='markers+lines', marker={'color': '#71c1e3'}, name='GPU / CodeCarbon'),
        go.Scatter(x=grouped.loc[0].index, y=grouped.loc[0,'externally_measured'], mode='markers+lines', marker={'color': '#009ee3'}, name='GPU / extern'),
        # go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'power_draw'], mode='markers+lines', marker={'color': '#e8a2c2'}, name='CPU / CodeCarbon'),
        # go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'externally_measured'], mode='markers+lines', marker={'color': '#e82e82'}, name='CPU / extern')
    ])
    # fig.update_yaxes(type="log")
    fig.update_layout(yaxis_title='Ws per inference', xaxis_title='Model', title='Profiling Comparison: CodeCarbon VS External Energy Meter (1/2)',
                    legend=dict(title='Processor / Profiling:', orientation="v", yanchor="top", y=0.95, xanchor="left", x=0.02),
                    margin={'l': 10, 'r': 10, 'b': 10, 't': 50}, width=900, height=600)
    fig.show()
    print(1)

    # plot correlation of parameters and energy difference
    # fig = go.Figure([
    #     go.Scatter(x=grouped.loc[0].index, y=grouped.loc[0,'diff'], mode='markers+lines', marker={'color': '#009ee3'}, name='GPU'),
    #     go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'diff'], mode='markers+lines', marker={'color': '#e82e82'}, name='CPU')
    # ])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=grouped.loc[0].index, y=grouped.loc[0,'diff'], mode='markers+lines', marker={'color': '#009ee3'}, name='GPU'), secondary_y=False)
    fig.add_trace(go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'diff'], mode='markers+lines', marker={'color': '#e82e82'}, name='CPU'), secondary_y=False)
    fig.add_trace(go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'parameters'], mode='markers+lines', marker={'color': '#35cdb4'}, name='Model Size'), secondary_y=True)

    fig.update_yaxes(type="log", title_text=r'$\Delta \text{Ws per inference}$', secondary_y=False)
    fig.update_yaxes(type="log", title_text=r'Number of Parameters', secondary_y=True)

    fig.update_layout(xaxis_title='Number of Parameters', title='Profiling Comparison: CodeCarbon VS External Energy Meter (2/2)',
                    legend=dict(orientation="h", yanchor="top", y=0.95, xanchor="left", x=0.02),
                    margin={'l': 10, 'r': 10, 'b': 10, 't': 50}, width=900, height=600)
    fig.show()

    # for gpu, s in zip ([0, 1], ['GPU', 'CPU']):
    #     sub_db = db[db['nogpu'] == gpu]
    #     fig.add_traces([
    #         # go.Scatter(x=db['parameters'], y=db['diff'], mode='markers', name='extern'),
    #         go.Scatter(x=sub_db['model'], y=sub_db['externally_measured'], mode='markers', name=f'{s} (extern)'),
    #         go.Scatter(x=sub_db['model'], y=sub_db['power_draw'], mode='markers', name=f'{s} (CodeCarbon)')
    #     ])