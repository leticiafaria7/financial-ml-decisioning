import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_descriptions = {
    "age": "Age",
    "job": "Type of job",
    "marital": "Marital status",
    "education": "Education level",
    "default": "Has credit in default",
    "housing": "Has housing loan",
    "loan": "Has personal loan",
    "contact": "Contact communication type",
    "month": "Last contact month of year",
    "day_of_week": "Last contact day of the week",
    "duration": "Last contact duration, in seconds",
    "campaign": "Number of contacts performed during this campaign and for this client",
    "pdays": "Number of days that passed after the client was last contacted from a previous campaign",
    "previous": "Number of contacts performed before this campaign and for this client",
    "poutcome": "Outcome of the previous marketing campaign",
    "emp_var_rate": "Employment variation rate - quarterly indicator",
    "cons_price_idx": "Consumer price index - monthly indicator",
    "cons_conf_idx": "Consumer confidence index - monthly indicator",
    "euribor3m": "Euribor 3 month rate - daily indicator",
    "nr_employed": "Number of employees - quarterly indicator",
    "y": "Whether the client subscribed to a term deposit"
}

months = {
    "jan": "01. jan",
    "feb": "02. feb",
    "mar": "03. mar",
    "apr": "04. apr",
    "may": "05. may",
    "jun": "06. jun",
    "jul": "07. jul",
    "aug": "08. aug",
    "sep": "09. sep",
    "oct": "10. oct",
    "nov": "11. nov",
    "dec": "12. dec"
}

days = {
    "mon": "1. mon",
    "tue": "2. tue",
    "wed": "3. wed",
    "thu": "4. thu",
    "fri": "5. fri"
}

def faixa_etaria(x):
    if x <= 25:
        return "Até 25 anos"
    if x <= 30:
        return "Entre 26 e 30 anos"
    if x <= 40:
        return "Entre 31 e 40 anos"
    if x <= 50:
        return "Entre 41 e 50 anos"
    if x <= 60:
        return "Entre 50 e 60 anos"
    else:
        return "Mais de 60 anos"


def sep_milhar(num, casas_decimais = 0):
    return f"{num:,.{casas_decimais}f}".replace(",", ".")

def miss_cat_pd(df_) -> pd.DataFrame:
    ''' 
    Função que recebe um dataframe e retorna outro dataframe com o percentual de missings e o número de categorias de cada coluna 
    
    Args:
        df_: base de dados
    
    Returns: 
        pd.DataFrame: dataframe com 3 colunas: nome da coluna, percentual de missings da coluna e número de categorias da coluna 
    ''' 

    # lista vazia para anexar os percentuais de missings 
    perc_missings = [] 
    
    # lista das colunas do dataframe 
    colunas = df_.columns.tolist() 
    
    # calcula o número de missings de cada coluna 
    for coluna in colunas: 
        perc_miss = df_[coluna].isna().mean() * 100 
        perc_missings.append(perc_miss) 
    
    # lista vazia para anexar o número de categorias 
    n_categorias = [] 
    
    # calcular o número de categorias de cada coluna 
    for coluna in colunas: 
        n_cat = df_[coluna].nunique() 
        n_categorias.append(n_cat) 
 
    # junta tudo em um dataframe 
    df = pd.DataFrame(data = {'coluna': colunas,
                              'perc_missings': perc_missings,
                              'n_categorias': n_categorias}) 
    
    return df

def cont_perc_categorias(df_, coluna):
    df_temp = df_.copy()
    df_temp = df_temp[coluna].value_counts(dropna = False).reset_index()
    df_temp['perc'] = round(df_temp['count'] * 100 / df_temp['count'].sum(), 3)

    return df_temp

def grafico_barras(df_, coluna, figsize = (15, 3), sort_categorias = False, casas_decimais = 1, espessura_barras = 0.6):

    tmp = df_[coluna].value_counts().reset_index()
    tmp['perc'] = tmp['count'] * 100 / tmp['count'].sum()

    if sort_categorias:
        tmp = tmp.sort_values(coluna)
        
    plt.figure(figsize = figsize)
    sns.set_style('white') # also 'ticks', 'darkgrid', 'whitegrid', white
    fig = sns.barplot(data = tmp, x = 'count', y = coluna, color = '#7f7f7f', width = espessura_barras)

    for i, (count, perc) in enumerate(zip(tmp['count'], tmp['perc'])):
        fig.annotate(f'{sep_milhar(count)} ({perc:.{casas_decimais}f}%)',
                    (count, i),
                    ha = 'left', va = 'center', fontsize = 10,
                    color = '#4d4d4d', xytext = (1, 0),
                    textcoords = 'offset points')

    # remover os números do eixo x
    fig.set_xticklabels('')
    fig.set_yticklabels(fig.get_yticklabels(), fontsize=10)

    # remover os títulos dos eixos
    fig.set(xlabel = '', ylabel = '')

    # remover as bordas do gráfico
    sns.despine(bottom = True)


import pandas as pd
import plotly.express as px


def grafico_yes_no(df, coluna_categoria, coluna_y="y"):
    """
    Cria um gráfico de barras empilhadas mostrando a distribuição
    de 'yes'/'no' dentro de cada categoria.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna_categoria : str
        Nome da coluna categórica usada no eixo Y.
    coluna_y : str, default="y"
        Nome da coluna contendo 'yes'/'no'.

    Retorna
    -------
    fig : plotly.graph_objects.Figure
        Figura Plotly.
    """

    # ---------------------------------------------------------
    # 1. Contagem absoluta
    # ---------------------------------------------------------
    dados = (
        df[[coluna_categoria, coluna_y]]
        .dropna()
        .groupby([coluna_categoria, coluna_y])
        .size()
        .reset_index(name="valor")
    )

    # Total por categoria
    dados["total"] = dados.groupby(coluna_categoria)["valor"].transform("sum")

    # Percentual dentro de cada categoria
    dados["percentual"] = dados["valor"] / dados["total"] * 100

    # Texto que aparecerá no centro das barras
    dados["rotulo"] = (
        dados["valor"].astype(str)
        + " ("
        + dados["percentual"].round(1).astype(str)
        + "%)"
    )

    # ---------------------------------------------------------
    # 2. Ordenação das categorias
    # ---------------------------------------------------------
    ordem_categorias = (
        dados.groupby(coluna_categoria)["total"]
        .first()
        .sort_values()
        .index
        .tolist()
    )

    # ---------------------------------------------------------
    # 3. Gráfico
    # ---------------------------------------------------------
    fig = px.bar(
        dados,
        x="percentual",
        y=coluna_categoria,
        color=coluna_y,
        orientation="h",
        text="rotulo",
        category_orders={
            coluna_categoria: ordem_categorias,
            coluna_y: ["no", "yes"]
        },
        color_discrete_map={
            "no": "#ac3a4e",
            "yes": "#356859"
        },
        custom_data=[
            "valor",
            "percentual",
            "total"
        ],
    )

    # ---------------------------------------------------------
    # 4. Rótulos no centro das barras
    # ---------------------------------------------------------
    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate=(
            f"<b>%{{y}}</b><br>"
            f"Resposta: %{{fullData.name}}<br>"
            f"Valor: %{{customdata[0]:,}}<br>"
            f"Percentual: %{{customdata[1]:.1f}}%<br>"
            f"Total da categoria: %{{customdata[2]:,}}"
            "<extra></extra>"
        ),
    )

    # ---------------------------------------------------------
    # 5. Altura dinâmica
    # ---------------------------------------------------------
    n_categorias = dados[coluna_categoria].nunique()

    # Aproximadamente 55 px por categoria
    altura = max(300, min(1200, n_categorias * 55))

    fig.update_layout(
        height=altura,
        barmode="stack",
        xaxis_title="Quantidade",
        yaxis_title=None,
        legend_title=None,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig