#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import networkx as nx
import plotly.graph_objs as go

import pandas as pd
from colour import Color
from datetime import datetime
from textwrap import dedent as d
import json
import subprocess
import numpy as np
import sys

# import the css template, and pass the css template into dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Visualize Knowledge Graph"

fname = "medical_notes"


##############################################################################################################################################################
def network_graph(use_noun_chunks, use_ner, spacy_model, language_model, entity_linker, threshold,
                  cmd="env/bin/python extract.py examples/IN_FILE OUT_FILE --use_cuda false --REL_embeddings_path REL_Experiments/wiki_2019/generated"):
    IN_FILE = f"{fname}.txt"
    OUT_FILE = f"{fname}-{language_model}.jsonl"
    OUT_FILE = OUT_FILE.replace("/", "-")
    cmd = cmd.replace("IN_FILE", IN_FILE).replace("OUT_FILE", OUT_FILE)

    if use_ner:
        cmd += " --use_ner true"
    else:
        cmd += " --use_ner false"
    if use_noun_chunks:
        cmd += " --use_nouns true"
    else:
        cmd += " --use_nouns false"

    cmd += f" --language_model {language_model} --spacy_model {spacy_model} --entity_linker {entity_linker} --threshold {threshold}"
    print(f"executing: {cmd}")

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    print(f"done executing: \n{cmd}, p.communicate: {p.communicate()}")
    G = nx.DiGraph()
    with open(OUT_FILE, 'r') as f:
        for l in f.readlines():
            current_line = json.loads(l)
            for tup in current_line['tri']:
                G.add_edge(tup['h'], tup['t'], label=str(tup['r']) + "_" + str(np.round(tup['c'], 2)))

    pos = nx.shell_layout(G)

    edge_x = []
    edge_y = []

    etext = [f'{w}' for w in list(nx.get_edge_attributes(G, 'label').values())]
    xtext = []
    ytext = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        xtext.append((x0 + x1) / 2)  # for edge text
        ytext.append((y0 + y1) / 2)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        mode='lines')

    eweights_trace = go.Scatter(x=xtext, y=ytext, mode='text',
                                marker_size=0.5,
                                text=etext,
                                textposition='top center',
                                hovertemplate='weight: %{text}<extra></extra>')

    node_x = []
    node_y = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition="bottom center",
        text=list(G.nodes()),
        marker={'size': 50, 'color': 'LightSkyBlue'}
    )

    figure = go.Figure(data=[edge_trace, node_trace, eweights_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                           xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                           yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                           height=700,
                           annotations=[
                               dict(
                                   ax=(pos[edge[0]][0] + pos[edge[1]][0]) / 2,
                                   ay=(pos[edge[0]][1] + pos[edge[1]][1]) / 2, axref='x', ayref='y',
                                   # x=(pos[edge[1]][0] * 3 + pos[edge[0]][0]) / 4,
                                   # y=(pos[edge[1]][1] * 3 + pos[edge[0]][1]) / 4, xref='x', yref='y',
                                   x=(pos[edge[1]][0] * 9 + pos[edge[0]][0]) / 10,
                                   y=(pos[edge[1]][1] * 9 + pos[edge[0]][1]) / 10, xref='x', yref='y',
                                   # ax=(pos[edge[0]][0] + pos[edge[1]][0]) / 2,
                                   # ay=(pos[edge[0]][1] + pos[edge[1]][1]) / 2, axref='x', ayref='y',
                                   # # x=(pos[edge[1]][0] * 3 + pos[edge[0]][0]) / 4,
                                   # # y=(pos[edge[1]][1] * 3 + pos[edge[0]][1]) / 4, xref='x', yref='y',
                                   # x=(pos[edge[1]][0] * 9 + pos[edge[0]][0]) / 10,
                                   # y=(pos[edge[1]][1] * 9 + pos[edge[0]][1]) / 10, xref='x', yref='y',
                                   showarrow=True,
                                   arrowhead=3,
                                   arrowsize=4,
                                   arrowwidth=1,
                                   opacity=1
                               ) for edge in G.edges]
                       ))
    return figure


######################################################################################################################################################################
# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([

    html.H1("Visualize Knowledge Graphs", style={'text-align': 'center'}),

    html.Div([
        html.Div(daq.BooleanSwitch(
            id='ner',
            label="Use NERs", labelPosition="right",

           #style={"width": 140, "margin-left": "15px", "margin-right": "15px", 'display': 'inline-block'},
            on=False,), style={'width': '10%', 'display': 'inline-block', "margin-left": "5px", "margin-right": "15px"}),

        html.Div(daq.BooleanSwitch(
            id='noun',
            label="Use Noun Chunks", labelPosition="right",
            # style={"width": 180, "margin-left": "15px", "margin-right": "15px", 'display': 'inline-block'},
            on=True
        ), style={'width': '15%', 'display': 'inline-block', "margin-right": "15px"}),
        html.Div(html.Label(["Language Model",
                    dcc.Dropdown(id="language_model",
                                 options=[
                                     {"label": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                                      "value": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"},
                                     {"label": "emilyalsentzer/Bio_ClinicalBERT",
                                      "value": "emilyalsentzer/Bio_ClinicalBERT"},
                                     {"label": "bert-large-uncased", "value": "bert-large-uncased"},
                                     {"label": "bert-large-cased", "value": 'bert-large-cased'},
                                     {"label": "roberta-large (cased)", "value": 'roberta-large'},
                                     {"label": "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16",
                                      "value": 'bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16'},
                                     {"label": "allenai/biomed_roberta_base",
                                      "value": "allenai/biomed_roberta_base"},
                                     {"label": 'bert-base-uncased', "value": 'bert-base-uncased'},
                                     {"label": "bert-base-cased", "value": 'bert-base-cased'},
                                     {"label": "gpt2", "value": 'gpt2'},
                                     {"label": "gpt2-medium", "value": 'gpt2-medium'},
                                     {"label": "gpt2-large", "value": 'gpt2-large'},
                                     {"label": "gpt2-xl", "value": 'gpt2-xl'},
                                 ],
                                 multi=False,
                                 value='bert-base-cased',

                                 )]), style={'width': '30%', 'display': 'inline-block',"margin-left": "15px", "margin-right": "15px"}),
        html.Div(html.Label(["Spacy Model",
                    dcc.Dropdown(id="spacy_model",
                                 options=[
                                     {"label": "en_core_sci_lg", "value": 'en_core_sci_lg'},
                                     {"label": "en_core_web_md", "value": "en_core_web_md"},
                                     {'label': 'en_ner_bc5cdr_md', 'value': 'en_ner_bc5cdr_md'},
                                 ],

                                 multi=False,
                                 value='en_core_web_md'
                                 )]
                   ), style={'width': '15%', 'display': 'inline-block',"margin-left": "15px", "margin-right": "15px"}),
        html.Div(html.Label(["Entity Linker",
                             dcc.Dropdown(id="entity_linker",
                                          options=[
                                              {"label": "REL", "value": 'REL'},
                                              {"label": "scispacy", "value": "SCISPACY"},
                                          ],
                                          multi=False,
                                          value='REL'
                                          )]
                            ), style={'width': '10%', 'display': 'inline-block',"margin-left": "15px", "margin-right": "15px"}),
        html.Div(html.Label(["Threshold",
                             dcc.Dropdown(id="threshold",
                                          options=[
                                              {"label": "0.1", "value": 0.1},
                                              {"label": "0.05", "value": 0.05},
                                              {"label": "0.003", "value": 0.003},
                                          ],
                                          multi=False,
                                          value=0.003
                                          )]
                            ), style={'width': '10%', 'display': 'inline-block',"margin-left": "15px", "margin-right": "15px"})
        ],
        style={'text-align':'center'}
    ),

    html.Div(dcc.Graph(id='my_graph', figure={})),
    html.Br(),
    html.Br(),
    html.Button('Submit', id='submit-val', n_clicks=0)

])

# Connect the Plotly graphs with Dash Components
@app.callback(
    dash.dependencies.Output(component_id='my_graph', component_property='figure'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    state=[dash.dependencies.State(component_id='noun', component_property='on'),
           dash.dependencies.State(component_id='ner', component_property='on'),
           dash.dependencies.State(component_id='spacy_model', component_property='value'),
           dash.dependencies.State(component_id='language_model', component_property='value'),
           dash.dependencies.State(component_id='entity_linker', component_property='value'),
           dash.dependencies.State(component_id='threshold', component_property='value'),
           ]
)
def update_graph(n_clicks, noun, ner, spacy_model, lang, entity_linker, threshold):
    print(f"noun: {noun}")
    print(f"ner: {ner}")
    print(f"lang: {lang}")
    print(f"spacy model: {spacy_model}")
    print(f"entity_linker: {entity_linker}")
    print(f"threshold: {threshold}")
    return network_graph(noun, ner, spacy_model, lang, entity_linker,threshold)


if __name__ == '__main__':
    app.run_server(debug=True)
