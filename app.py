#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import statsmodels.api as sm
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')

#-------------------------------------
# Intial Settings
#-------------------------------------

# デフォルトでワイド表示
# streamlitのコードの先頭でなければならない
st.set_page_config(page_title='Easy Data Analyser', page_icon='./image/icon.png', layout='wide')

# 初回ロード時のみ風船を飛ばす
if 'is_first_load' not in st.session_state:
    st.session_state['is_first_load'] = True

if st.session_state['is_first_load']:
    st.session_state['is_first_load'] = False
    st.balloons()

#-------------------------------------
# Sidebar
#-------------------------------------

st.sidebar.image(Image.open('./image/logo_medium.png'), use_column_width=False)
st.sidebar.header('設定')

# data = st.sidebar.file_uploader('CSVファイルのアップロード')
st.sidebar.warning('CSVアップロード機能は現在利用不可です。サンプルデータを利用してください。')
data = None
if data is not None:
    df = pd.read_csv(data, encoding='utf-8')
else:
    if st.sidebar.checkbox('サンプルデータを使用'):
        sample_data_name = st.sidebar.selectbox('サンプルデータを選択', ('ボストン住宅価格', 'PSIプログラム'))
        if sample_data_name == 'ボストン住宅価格':
            data = load_boston()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['PRICE'] = data.target
        elif sample_data_name == 'PSIプログラム':
            data = sm.datasets.spector.load_pandas()
            df = pd.DataFrame(data.data)
    else:
        df = None

if df is not None:
    if st.sidebar.checkbox('時系列設定'):
        is_datetime = True
        date_col = st.sidebar.selectbox('時系列を選択してください', tuple(df.columns))
        df.index = pd.to_datetime(df[date_col])
        df.drop(columns=date_col, inplace=True)
    else:
        is_datetime = False
else:
    pass

#-------------------------------------
# Main
#-------------------------------------

st.image(Image.open('./image/logo_large.png'), use_column_width=False)
if df is not None:
    method = st.sidebar.selectbox('分析手法', ('データの確認', '相関分析', '回帰分析'))

    if method == 'データの確認':
        st.header('データの確認')
        st.dataframe(df)

        st.header('要約統計量')
        st.dataframe(df.describe())

        st.header('欠損値の確認')
        st.dataframe(df.isnull().sum())

    if method == '相関分析':
        st.header('相関分析')
        st.dataframe(df.corr())

    if method == '回帰分析':
        st.header('回帰分析')

        reg_type = st.selectbox('タイプ', ('線形回帰', 'ロジスティック回帰'))
        families = {
            '線形回帰': None,
            'ロジスティック回帰': sm.families.Binomial()
        }
        family = families[reg_type]

        """
        with st.expander('詳細'):
            if reg_type == '線形回帰':
                with open('./markdown/regression.md') as f:
                    md = f.read()
                st.markdown(md)
            elif reg_type == 'ロジスティック回帰':
                with open('./markdown/logistic_regression.md') as f:
                    md = f.read()
                st.markdown(md)
        """

        y = st.selectbox('目的変数を選択してください', tuple(df.columns))

        if is_datetime:
            y_plot = go.Figure([go.Scatter(x=df.index, y=df[y])])
            st.plotly_chart(y_plot, use_container_width=True)
        else:
            y_hist = go.Figure([go.Histogram(x=df[y], nbinsx=int(np.round(1+np.log2(len(df[y])))))])
            st.plotly_chart(y_hist, use_container_width=True)

        without_y = list(df.columns)
        without_y.remove(y)
        X = st.multiselect('説明変数を選択してください(複数選択可)', tuple(without_y), default=tuple(without_y))

        if X:
            if st.button(label='実行', key='is_exec'):
                if family is not None:
                    reg_model = sm.GLM(df[y], sm.add_constant(df[X]), family=family).fit()
                else:
                    reg_model = sm.OLS(df[y], sm.add_constant(df[X])).fit()
                st.text(reg_model.summary().as_text())
else:
    with open('./markdown/main.md') as f:
        md = f.read()
    st.markdown(md)
