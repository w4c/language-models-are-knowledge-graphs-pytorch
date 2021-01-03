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


# import the css template, and pass the css template into dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Visualize Knowledge Graph"


fname = "medical_notes"



##############################################################################################################################################################
def network_graph(use_noun_chunks, use_ner, language_model, cmd="env/bin/python extract.py examples/IN_FILE OUT_FILE --use_cuda false --REL_embeddings_path REL_Experiments/wiki_2019/generated"):
    IN_FILE = f"{fname}.txt"
    OUT_FILE = f"{fname}-{language_model}.jsonl"
    cmd = cmd.replace("IN_FILE", IN_FILE).replace("OUT_FILE", OUT_FILE)

    if use_ner:
        cmd += " --use_ner true"
    if use_noun_chunks:
        cmd += " --use_nouns true"

    cmd += f" --language_model {language_model}"
    print(cmd)

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    print("done ", p.communicate())
    G = nx.DiGraph()
    with open(OUT_FILE, 'r') as f:
        for l in f.readlines():
            current_line = json.loads(l)
            for tup in current_line['tri']:
                G.add_edge(tup['h'], tup['t'], confidence=str(tup['r']) + "_" + str(np.round(tup['c'],2)))


    # G = nx.Graph()
    # G.add_node( 'a' )
    # G.add_node( 'b' )
    # G.add_node( 'c' )
    # G.add_node( 'd' )
    # G.add_edge( 'a', 'b' )
    # G.add_edge( 'c', 'd' )


    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

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
        marker={'size': 50, 'color': 'LightSkyBlue'}
    )

    figure =  go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
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
    daq.BooleanSwitch(
        id='noun',
        label="Use Noun Chunks",
        on=True
    ),
    html.Br(),
    daq.BooleanSwitch(
        id='ner',
        label="Use NERs",
        on=False
    ),

    dcc.Dropdown(id="language_model",
                  options=[
                      {"label": "bert-large-uncased", "value": "bert-large-uncased"},
                      {"label": "bert-large-cased", "value": 'bert-large-cased'},
                      {"label": 'bert-base-uncased', "value": 'bert-base-uncased'},
                      {"label": "bert-base-cased", "value": 'bert-base-cased'},
                      {"label": "gpt2", "value": 'gpt2'},
                      {"label": "gpt2-medium", "value": 'gpt2-medium'},
                      {"label": "gpt2-large", "value": 'gpt2-large'},
                      {"label": "gpt2-xl", "value": 'gpt2-xl'},
                      {"label": "bio-bert", "value": 'bio-bert'},
                          ],
                  multi=False,
                  value='bert-large-cased',
                  style={'width': "100%"}
                  ),


    html.Br(),

    dcc.Graph(id='my_graph', figure={})

])
# Connect the Plotly graphs with Dash Components
@app.callback(
    dash.dependencies.Output(component_id='my_graph', component_property='figure'),
    [dash.dependencies.Input(component_id='noun', component_property='on'),
     dash.dependencies.Input(component_id='ner', component_property='on'),
     dash.dependencies.Input(component_id='language_model', component_property='value')
     ]
)
def update_graph(noun, ner, lang):
    print("calling ..")
    return network_graph(noun,ner,lang)


if __name__ == '__main__':
    app.run_server(debug=True)
