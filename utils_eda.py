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
    "emp.var.rate": "Employment variation rate - quarterly indicator",
    "cons.price.idx": "Consumer price index - monthly indicator",
    "cons.conf.idx": "Consumer confidence index - monthly indicator",
    "euribor3m": "Euribor 3 month rate - daily indicator",
    "nr.employed": "Number of employees - quarterly indicator",
    "y": "Whether the client subscribed to a term deposit"
}

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
        fig.annotate(f'{int(count)} ({perc:.{casas_decimais}f}%)',
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
    