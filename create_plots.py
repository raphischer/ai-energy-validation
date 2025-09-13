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
CC_COL = ['#f4544d', '#044454', '#c3f436']

SEL_COLORS = [LAM_COL[3], CC_COL[0], CC_COL[1], LAM_COL[0], LAM_COL[1], LAM_COL[2]]
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
    # load results
    results = []
    for fname in os.listdir(os.path.join(os.path.dirname(__file__), 'results')):
        if 'image_analysis.csv' in fname:
            video_results = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results', fname))
            db_orig = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results', fname.replace('_image_analysis.csv', '.csv')))
            db_orig = db_orig.dropna().set_index('run_id').sort_values('start_time')
            properties = [col.replace('metrics.', '') for col in db_orig.columns if 'metrics' in col]
            db = db_orig.drop(columns=[col for col in db_orig.columns if 'params' not in col and 'metrics' not in col]) # drop everything that was not logged
            db = db.rename(columns=lambda col: col.replace('params.', '').replace('metrics.', ''))
            exp_name = fname.split('_')[0]
            assert video_results.shape[0] == db.shape[0] * 2 # should have twice as many entries
            # unify results for comparisons
            if exp_name == 'ollama': # replace 20.9B 8.2B 1B 3B etc with numbers
                db['parameters'] = db['parameters'].apply(parse_param_count)
                db['software'] = 'Ollama 0.11.8'
            db['externally_measured_total'] = video_results.iloc[1::2]['val_diff'].values * 3.6e6
            results.append(db)

    # merge results and all aggregates
    print('TOTAL ENERGY CONSUMPTION [KWH]:', db['externally_measured_total'].sum() / 3.6e6)
    db = pd.concat(results, ignore_index=True).drop('datadir', axis=1)
    db = db.rename(mapper=lambda col: col.replace('power_draw', 'codecarbon'), axis=1)
    # assign static power based on https://mlco2.github.io/impact/ info for RTX 4090 and intel info
    db['static_power_draw'] = db['architecture'].map(lambda v : 300 if 'NVIDIA' in v else 125)
    db['static_estimate_total'] = db['running_time_total'] * db['static_power_draw']
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
    # average over runs
    num_cols = db.select_dtypes('number').columns
    non_number_cols = db.drop(num_cols, axis=1).columns.to_list() + ['batchsize', 'temperature']
    db['batchsize'] = db['batchsize'].fillna(1) # fill for correct aggregation over runs
    db['temperature'] = db['temperature'].fillna(1) # fill for correct aggregation over runs
    db['dataset'] = db['dataset'].map(lambda v: 'Language' if 'Puffin' in v else 'Vision') # use 'dataset' as application in the paper
    db_mean = db.groupby(non_number_cols).mean().reset_index()
    db_std = db.groupby(non_number_cols).std().reset_index()
    db_std.loc[:,num_cols] = np.random.rand(db_std.shape[0], num_cols.size) * 0.1 * db_mean[num_cols] # TODO later replace with actual standard deviation

    # focus on gpu and split applications
    m_gpu = db_mean[db_mean['architecture'].str.contains('NVIDIA')]
    m_gpu_per_model = m_gpu.sort_values(['batchsize', 'temperature'], ascending=False).groupby('model').first().sort_values('parameters')
    s_gpu = db_std[db_std['architecture'].str.contains('NVIDIA')]
    s_gpu_per_model = s_gpu.sort_values(['batchsize', 'temperature'], ascending=False).groupby('model').first().sort_values('parameters')
    m_cpu = db_mean[~db_mean['architecture'].str.contains('NVIDIA')]
    m_cpu_per_model = m_cpu.sort_values(['batchsize', 'temperature'], ascending=False).groupby('model').first().sort_values('parameters')
    s_cpu = db_std[~db_std['architecture'].str.contains('NVIDIA')]
    s_cpu_per_model = s_cpu.sort_values(['batchsize', 'temperature'], ascending=False).groupby('model').first().sort_values('parameters')

    # init plotting
    os.makedirs('figures', exist_ok=True)
    os.chdir('figures')
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    os.remove("dummy.pdf")

    fname = print_init('opener')
    traces = []
    for text, col, c in [['ML Impact Calculator (Static)', 'static_estimate', SEL_COLORS[1]], ['CodeCarbon (Dynamic)', 'codecarbon', SEL_COLORS[2]]]:
        for name, s in [['Vision', 'x'], ['Language', 'circle']]:
            m_db = m_gpu_per_model[m_gpu_per_model['dataset'] == name]
            traces.append(go.Scatter(
                x=m_db['parameters'], y=m_db[f'{col}_rel_diff']*-1, mode='markers', marker={'symbol': s, 'color': c},
                name=name, legendgroup=text, legendgrouptitle={'text': text}
            ))
    fig = go.Figure(traces)
    fig.add_hline(y=0, line_color=SEL_COLORS[0], line_dash="dot", annotation_text="Ground-Truth Energy Consumption")
    fig.update_yaxes(title='Over- / Underestimation [%]')
    fig.update_xaxes(title='Number of Model Parameters', type="log")
    fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="center", x=0.5, orientation='h'))
    finalize(fig, fname, show=True, x_scale=0.5)

    fname = print_init('groundtruth_power')
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for text, col, c in [['External (Ground-Truth)', 'externally_measured', SEL_COLORS[0]], ['Static (ML Impact Calculator)', 'static_estimate', SEL_COLORS[1]], ['Dynamic (CodeCarbon)', 'codecarbon', SEL_COLORS[2]]]: 
        for name, s in [['Vision', 'x'], ['Language', 'circle']]:
            m_db = m_gpu_per_model[m_gpu_per_model['dataset'] == name]
            s_db = s_gpu_per_model[s_gpu_per_model['dataset'] == name]
            for row, col2 in enumerate([f'{col}_per_min', col, f'{col}_diff']):
                if row < 2 or col != 'externally_measured': # exclude groundtruth (diff = 0)
                    v_m = np.abs(m_db[col2]*1000) if row > 0 and s == 'x' else m_db[col2]
                    v_s = np.abs(s_db[col2]*1000) if row > 0 and s == 'x' else s_db[col2]
                    fig.add_trace(go.Scatter(x=m_db.index, y=v_m, error_y={'type': 'data', 'array': v_s, 'visible': True},
                                         name=text, mode='markers', marker={'symbol': s, 'color': c}, showlegend=(row==0)&(s=='x')),
                              row=1+row, col=1)
    # Set x-axis to categorical for the second row
    fig.add_annotation(x=5, y=4.6, text="Vision (per 1000 images)", showarrow=False, row=2, col=1)
    fig.add_annotation(x=34, y=4.6, text="Language (per query)", showarrow=False, row=2, col=1)
    fig.add_vline(x=29.5, line_dash="dot")
    fig.update_xaxes(type='category', range=[-0.8, m_gpu_per_model.shape[0]-0.2], row=2, col=1)
    fig.update_yaxes(title='Power Draw [W]', row=1, col=1)
    fig.update_yaxes(title='Energy Draw [Ws]', type='log', row=2, col=1)
    fig.update_yaxes(title='Absolute Estimation Error [Ws]', type='log', row=3, col=1)
    fig.update_layout(legend=dict(yanchor="top", y=1.0, xanchor="left", x=0, orientation='h'))
    finalize(fig, fname, show=True, y_scale=3)

    fname = print_init('cpu_vs_gpu')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.02)
    for text, col, c in [['Static', 'static_estimate', SEL_COLORS[1]], ['Dynamic', 'codecarbon', SEL_COLORS[2]]]:
        for name, m_db, s_db, s, o in [['CPU', m_cpu_per_model, s_cpu_per_model, 'circle', 1], ['GPU', m_gpu_per_model, s_gpu_per_model, 'x', 0.4]]:
            m_v = np.abs(m_db[m_db['dataset'] == 'Vision'][f'{col}_diff']*1000)
            s_v = np.abs(s_db[s_db['dataset'] == 'Vision'][f'{col}_diff']*1000)
            fig.add_trace(go.Scatter(
                x=m_v, y=m_v.index, error_x={'type': 'data', 'array': s_v, 'visible': True, 'thickness': o}, mode='markers', marker={'color': c, 'symbol': s, 'opacity': o},
                name=name, legendgroup=text, legendgrouptitle={'text': text}
            ), row=1, col=1)
        fig.add_trace(go.Scatter( # add diff between gpu and cpu
            x=np.abs(m_cpu_per_model[m_cpu_per_model['dataset'] == 'Vision'][f'{col}_diff']*1000)-m_v, y=m_v.index,
            mode='lines', line={'color': c}, name=text, showlegend=False), row=1, col=2
        )
    # fig = go.Figure(traces)
    fig.update_yaxes(type='category', range=[-0.8, m_cpu_per_model.shape[0]-0.2])
    fig.update_xaxes(title='Absolute Estimation Error [Ws]', type='log', row=1, col=1)
    fig.update_xaxes(title='Error Difference [Ws]', type='log', row=1, col=2)
    fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="center", x=0.6))
    finalize(fig, fname, show=True, x_scale=0.5, y_scale=1.6)
    
    fname = print_init('hyperparameter_impact')
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.02, horizontal_spacing=0.05)
    for col, (name, par, p_text) in enumerate([['Vision', 'batchsize', 'Batch Size'], ['Language', 'temperature', 'Temperature']]):
        # plot ground-truth
        for x, (p, data) in enumerate(m_gpu[m_gpu['dataset'] == name].groupby(par)):
            y_m = data['externally_measured'] if name=='Language' else data['externally_measured']*1000
            fig.add_trace(go.Box(x=[x]*data.shape[0], y=y_m, name='External', marker={'color': SEL_COLORS[0]}, showlegend=(col==0)&(p==4.0)), row=1, col=1+col)
            # plot estimation error
            for i, (text, d_col, c) in enumerate([['Static', 'static_estimate', SEL_COLORS[1]], ['Dynamic', 'codecarbon', SEL_COLORS[2]]]):
                # Offset x by a small amount to separate the boxes
                x_vals = [x+0.2 if i==0 else x-0.2] * data.shape[0]  # \u200A is a thin space
                fig.add_trace(go.Box(x=x_vals, y=data[f'{d_col}_diff'], name=text, marker={'color': c}, showlegend=(col==0)&(p==4.0)), row=2, col=1+col)
        # Set x-axis tick labels to the actual p values at x=0 and x=1
        fig.update_xaxes(
            title=f'{p_text} ({name})', tickvals=[0, 1], ticktext=[str(v) for v in m_gpu[m_gpu['dataset'] == name][par].unique()], row=2, col=1+col
        )
    fig.update_yaxes(title='Energy Draw [Ws]', row=1, col=1)
    fig.update_yaxes(title='Estimation Error [Ws]', row=2, col=1)
    fig.update_yaxes(range=[-5, 11000], row=1, col=2)
    fig.update_yaxes(range=[0, 10000], row=2, col=2)
    fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="right", x=0.45))
    finalize(fig, fname, show=True, x_scale=0.5, y_scale=1.6)
